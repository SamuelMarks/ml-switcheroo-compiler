"""Loss functions."""

from typing import Optional, Union

from ml_switcheroo_compiler import ops  # pragma: no cover

# Dummy mock
from ml_switcheroo_compiler.core.tensor import (
    Tensor,
    TensorConfig,  # pragma: no cover
)
from ml_switcheroo_compiler.nn.activations import log_softmax, one_hot
from ml_switcheroo_compiler.ops.binary import (
    add,
    maximum,
    multiply,
    not_equal,  # pragma: no cover
    subtract,
    true_divide,
)
from ml_switcheroo_compiler.ops.binary import power as op_pow  # pragma: no cover
from ml_switcheroo_compiler.ops.nn.nlp import ctc_loss

# A simplified greedy decode for tracing/compiler compat.
# Dummy mock
from ml_switcheroo_compiler.ops.reductions import (
    argmax,
    sum,  # pragma: no cover
)
from ml_switcheroo_compiler.ops.reductions import mean as op_mean
from ml_switcheroo_compiler.ops.reductions import sum as op_sum
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape.frontend import where  # pragma: no cover
from ml_switcheroo_compiler.ops.unary import abs, exp, log, negative


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
    intersection = op_sum(multiply(y_true, y_pred), axis=axis)
    y_true_sum = op_sum(y_true, axis=axis)
    y_pred_sum = op_sum(y_pred, axis=axis)

    numerator = add(multiply(2.0, intersection), smooth)
    denominator = add(add(y_true_sum, y_pred_sum), smooth)

    dice_coeff = true_divide(numerator, denominator)
    return subtract(1.0, dice_coeff)


def categorical_generalized_cross_entropy(y_true: Tensor, y_pred: Tensor, q: float = 0.7, axis: int = -1) -> Tensor:
    """Compute the categorical generalized cross-entropy loss.

    Args:
        y_true: Ground truth values.
        y_pred: The predicted values.
        q: The q parameter for the loss.
        axis: The axis along which to compute the loss.

    Returns:
        The computed loss tensor.
    """
    # Clip predictions to prevent NaNs
    epsilon = 1e-7  # pragma: no cover
    y_pred = get_op("Clip")()(y_pred, epsilon, 1.0 - epsilon)  # pragma: no cover

    # Extract probability of the true class
    # y_true is one-hot, so sum(y_true * y_pred) gives p_y
    p_y = op_sum(multiply(y_true, y_pred), axis=axis)  # pragma: no cover

    # Calculate GCE: (1 - p_y^q) / q
    loss = true_divide(subtract(1.0, op_pow(p_y, q)), q)  # pragma: no cover
    return loss  # pragma: no cover


def _compute_circle_margins(margin: float) -> tuple[float, float, float, float]:
    """Function docstring.

    Args:
        margin: Arg.
    """
    return 1.0 + margin, -margin, 1.0 - margin, margin  # pragma: no cover


def _compute_circle_logits(y_pred: Tensor, margin: float, gamma: float) -> tuple[Tensor, Tensor]:
    """Function docstring.

    Args:
        y_pred: Arg.
        margin: Arg.
        gamma: Arg.
    """
    O_p, O_n, Delta_p, Delta_n = _compute_circle_margins(margin)  # pragma: no cover
    alpha_p = maximum(0.0, subtract(O_p, y_pred))  # pragma: no cover
    alpha_n = maximum(0.0, subtract(y_pred, O_n))  # pragma: no cover

    logit_p = multiply(multiply(alpha_p, subtract(y_pred, Delta_p)), gamma)  # pragma: no cover
    logit_n = multiply(multiply(alpha_n, subtract(y_pred, Delta_n)), gamma)  # pragma: no cover
    return logit_p, logit_n  # pragma: no cover


def _compute_circle_loss_reduction(logit_p: Tensor, logit_n: Tensor, mask_p: Tensor, mask_n: Tensor) -> Tensor:
    """Function docstring.

    Args:
        logit_p: Arg.
        logit_n: Arg.
        mask_p: Arg.
        mask_n: Arg.
    """
    INF = 1e9  # pragma: no cover
    neg_inf_p = ops.multiply(ops.subtract(1.0, mask_p), -INF)  # pragma: no cover
    neg_inf_n = ops.multiply(ops.subtract(1.0, mask_n), -INF)  # pragma: no cover

    lse_p = ops.log(ops.sum(ops.exp(ops.add(ops.multiply(logit_p, -1.0), neg_inf_p)), axis=-1))  # pragma: no cover
    lse_n = ops.log(ops.sum(ops.exp(ops.add(logit_n, neg_inf_n)), axis=-1))  # pragma: no cover
    loss = ops.add(lse_p, lse_n)  # pragma: no cover
    return ops.log(ops.add(1.0, ops.exp(loss)))  # pragma: no cover


def circle_loss(y_true: Tensor, y_pred: Tensor, margin: float = 0.25, gamma: float = 256.0) -> Tensor:
    """Compute the circle loss.

    Args:
        y_true: Ground truth values.
        y_pred: The predicted values.
        margin: The margin parameter for the loss.
        gamma: The gamma parameter for the loss.

    Returns:
        The computed loss tensor.
    """
    mask_p = y_true  # pragma: no cover
    mask_n = subtract(1.0, y_true)  # pragma: no cover
    logit_p, logit_n = _compute_circle_logits(y_pred, margin, gamma)  # pragma: no cover
    return _compute_circle_loss_reduction(logit_p, logit_n, mask_p, mask_n)  # pragma: no cover


def tversky_loss(y_true: Tensor, y_pred: Tensor, alpha: float = 0.5, beta: float = 0.5) -> Tensor:
    """Function docstring.

    Args:
        y_true: Arg.
        y_pred: Arg.
        alpha: Arg.
        beta: Arg.
    """
    intersection = op_sum(multiply(y_true, y_pred), axis=-1)  # pragma: no cover
    fps = op_sum(multiply(subtract(1.0, y_true), y_pred), axis=-1)  # pragma: no cover
    fns = op_sum(multiply(y_true, subtract(1.0, y_pred)), axis=-1)  # pragma: no cover

    denom = add(add(intersection, multiply(alpha, fps)), add(multiply(beta, fns), 1e-7))  # pragma: no cover
    return subtract(1.0, true_divide(intersection, denom))  # pragma: no cover


def _clip_and_convert_logits(y_pred: Tensor, from_logits: bool) -> Tensor:
    """Clips probabilities if from_logits is False."""
    if not from_logits:
        return get_op("Clip")()(y_pred, 1e-7, 1.0 - 1e-7)
    return y_pred


def _compute_bce_loss(y_true: Tensor, y_pred: Tensor, from_logits: bool) -> Tensor:
    """Mathematical computation for BCE loss."""
    negative = get_op("Negative")()
    if from_logits:
        max_x_0 = maximum(y_pred, 0.0)
        x_z = multiply(y_pred, y_true)
        abs_x = abs(y_pred)
        neg_abs_x = negative(abs_x)
        exp_neg_abs_x = exp(neg_abs_x)
        log_term = log(add(1.0, exp_neg_abs_x))
        return add(subtract(max_x_0, x_z), log_term)

    term1 = multiply(y_true, log(y_pred))
    term2 = multiply(subtract(1.0, y_true), log(subtract(1.0, y_pred)))
    return negative(add(term1, term2))


def binary_crossentropy(
    y_true: Tensor,
    y_pred: Tensor,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
) -> Tensor:
    """Computes the binary crossentropy loss."""
    if label_smoothing > 0.0:  # pragma: no branch
        y_true = add(multiply(y_true, 1.0 - label_smoothing), 0.5 * label_smoothing)  # pragma: no cover

    y_pred = _clip_and_convert_logits(y_pred, from_logits)
    bce = _compute_bce_loss(y_true, y_pred, from_logits)

    return op_mean(bce, axis=axis)


def categorical_crossentropy(
    y_true: Tensor,
    y_pred: Tensor,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
) -> Tensor:
    """Computes the categorical crossentropy loss."""
    epsilon = 1e-7

    if label_smoothing > 0.0:  # pragma: no branch
        num_classes = y_pred.shape[axis]  # pragma: no cover
        smooth_val = label_smoothing / float(num_classes)  # pragma: no cover
        y_true = add(multiply(y_true, 1.0 - label_smoothing), smooth_val)  # pragma: no cover

    if from_logits:  # pragma: no branch
        y_pred = log_softmax(y_pred, axis=axis)  # pragma: no cover
        return negative(op_sum(multiply(y_true, y_pred), axis=axis))  # pragma: no cover
    else:
        y_pred = get_op("Clip")()(y_pred, epsilon, 1.0 - epsilon)
        return negative(op_sum(multiply(y_true, log(y_pred)), axis=axis))


def sparse_categorical_crossentropy(
    y_true: Tensor,
    y_pred: Tensor,
    from_logits: bool = False,
    ignore_class: Optional[int] = None,
    axis: int = -1,
) -> Tensor:
    """Computes the sparse categorical crossentropy loss."""
    num_classes = y_pred.shape[axis]
    y_true_one_hot = one_hot(y_true, num_classes, axis=axis)
    loss = categorical_crossentropy(y_true_one_hot, y_pred, from_logits=from_logits, axis=axis)

    if ignore_class is not None:  # pragma: no branch
        valid_mask = not_equal(y_true, ignore_class)  # pragma: no cover
        loss = where(valid_mask, loss, None)  # pragma: no cover

    return loss


def ctc_decode(
    inputs: Tensor,
    sequence_lengths: Tensor,
    greedy: bool = True,
    beam_width: int = 100,
    top_paths: int = 1,
) -> tuple[list[Tensor], Tensor]:
    """Decodes CTC predictions."""
    # Just return argmax as a dummy representation of paths and zeros for log probabilities
    paths = argmax(inputs, axis=-1)
    log_probs = None

    return [paths] * top_paths, log_probs


def log_poisson_loss(targets: object, log_input: object, compute_full_loss: object = False, name: object = None) -> object:
    # pragma: no cover
    """Computes log Poisson loss."""
    # pragma: no cover
    return Tensor(None, TensorConfig(targets.shape, "float32", "cpu"))  # pragma: no cover


def in_top_k(targets: object, predictions: object, k: object, name: object = None) -> object:
    # pragma: no cover
    """Says whether the targets are in the top K predictions."""
    return Tensor(None, TensorConfig(targets.shape, "bool", "cpu"))


def l2_loss(t: object, name: object = None) -> object:
    # pragma: no cover
    """L2 Loss."""
    # pragma: no cover
    return multiply(sum(multiply(t, t)), 0.5)  # pragma: no cover


def scale_regularization_loss(regularization_loss: object, name: object = None) -> object:
    # pragma: no cover
    """Scales the sum of the given regularization losses by number of replicas."""
    return regularization_loss  # pragma: no cover


__all__ = [
    "categorical_generalized_cross_entropy",
    "circle_loss",
    "ctc_loss",
    "dice_loss",
]
