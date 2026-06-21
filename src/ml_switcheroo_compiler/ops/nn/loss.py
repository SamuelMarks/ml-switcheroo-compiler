"""Loss functions."""

from typing import Optional, Union

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.reductions.frontend import ctc_loss


def dice_loss(
    y_true: Tensor,
    y_pred: Tensor,
    axis: Optional[Union[tuple[int, ...], int]] = None,
    smooth: float = 1e-5,
) -> Tensor:
    """Computes the Dice loss.

    Args:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predictions.
        axis (Optional[Union[tuple[int, ...], int]]): The axes to reduce over.
        smooth (float): Smoothing factor.

    Returns:
        Tensor: The calculated dice loss.
    """
    from ml_switcheroo_compiler.ops.binary import add, multiply, subtract, true_divide
    from ml_switcheroo_compiler.ops.reductions import sum as op_sum

    intersection = op_sum(multiply(y_true, y_pred), axis=axis)
    y_true_sum = op_sum(y_true, axis=axis)
    y_pred_sum = op_sum(y_pred, axis=axis)

    numerator = add(multiply(2.0, intersection), smooth)
    denominator = add(add(y_true_sum, y_pred_sum), smooth)

    dice_coeff = true_divide(numerator, denominator)
    return subtract(1.0, dice_coeff)


__all__ = ["ctc_loss", "dice_loss", "categorical_generalized_cross_entropy", "circle_loss"]


def categorical_generalized_cross_entropy(
    y_true: Tensor, y_pred: Tensor, q: float = 0.7, axis: int = -1
) -> Tensor:
    """Compute the categorical generalized cross-entropy loss.

    Args:
        y_true: Ground truth values.
        y_pred: The predicted values.
        q: The q parameter for the loss.
        axis: The axis along which to compute the loss.

    Returns:
        The computed loss tensor.
    """
    from ml_switcheroo_compiler.ops.binary import multiply, subtract, true_divide
    from ml_switcheroo_compiler.ops.aliases.math_ops import clip
    from ml_switcheroo_compiler.ops.reductions import sum as op_sum
    from ml_switcheroo_compiler.ops.binary import power as op_pow

    # Clip predictions to prevent NaNs
    epsilon = 1e-7
    y_pred = clip(y_pred, a_min=epsilon, a_max=1.0 - epsilon)

    # Extract probability of the true class
    # y_true is one-hot, so sum(y_true * y_pred) gives p_y
    p_y = op_sum(multiply(y_true, y_pred), axis=axis)

    # Calculate GCE: (1 - p_y^q) / q
    loss = true_divide(subtract(1.0, op_pow(p_y, q)), q)
    return loss


def _compute_circle_margins(margin: float) -> tuple[float, float, float, float]:
    return 1.0 + margin, -margin, 1.0 - margin, margin


def _compute_circle_logits(y_pred: Tensor, margin: float, gamma: float) -> tuple[Tensor, Tensor]:
    from ml_switcheroo_compiler.ops.binary import multiply, subtract
    from ml_switcheroo_compiler.ops.binary import maximum

    O_p, O_n, Delta_p, Delta_n = _compute_circle_margins(margin)
    alpha_p = maximum(0.0, subtract(O_p, y_pred))
    alpha_n = maximum(0.0, subtract(y_pred, O_n))

    logit_p = multiply(multiply(alpha_p, subtract(y_pred, Delta_p)), gamma)
    logit_n = multiply(multiply(alpha_n, subtract(y_pred, Delta_n)), gamma)
    return logit_p, logit_n


def _compute_circle_loss_reduction(  # pylint: disable=too-many-locals
    logit_p: Tensor, logit_n: Tensor, mask_p: Tensor, mask_n: Tensor
) -> Tensor:
    from ml_switcheroo_compiler.ops.binary import multiply, subtract, add
    from ml_switcheroo_compiler.ops.reductions import sum as op_sum
    from ml_switcheroo_compiler.ops.unary import exp, log

    INF = 1e9
    neg_inf_p = multiply(subtract(1.0, mask_p), -INF)
    neg_inf_n = multiply(subtract(1.0, mask_n), -INF)

    lse_p = log(op_sum(exp(add(multiply(logit_p, -1.0), neg_inf_p)), axis=-1))
    lse_n = log(op_sum(exp(add(logit_n, neg_inf_n)), axis=-1))
    loss = add(lse_p, lse_n)
    return log(add(1.0, exp(loss)))


def circle_loss(
    y_true: Tensor, y_pred: Tensor, margin: float = 0.25, gamma: float = 256.0
) -> Tensor:
    """Compute the circle loss.

    Args:
        y_true: Ground truth values.
        y_pred: The predicted values.
        margin: The margin parameter for the loss.
        gamma: The gamma parameter for the loss.

    Returns:
        The computed loss tensor.
    """
    from ml_switcheroo_compiler.ops.binary import subtract

    mask_p = y_true
    mask_n = subtract(1.0, y_true)
    logit_p, logit_n = _compute_circle_logits(y_pred, margin, gamma)
    return _compute_circle_loss_reduction(logit_p, logit_n, mask_p, mask_n)


def tversky_loss(y_true: Tensor, y_pred: Tensor, alpha: float = 0.5, beta: float = 0.5) -> Tensor:
    from ml_switcheroo_compiler.ops.binary import multiply, subtract, add, true_divide
    from ml_switcheroo_compiler.ops.reductions import sum as op_sum

    intersection = op_sum(multiply(y_true, y_pred), axis=-1)
    fps = op_sum(multiply(subtract(1.0, y_true), y_pred), axis=-1)
    fns = op_sum(multiply(y_true, subtract(1.0, y_pred)), axis=-1)

    denom = add(add(intersection, multiply(alpha, fps)), add(multiply(beta, fns), 1e-7))
    return subtract(1.0, true_divide(intersection, denom))
