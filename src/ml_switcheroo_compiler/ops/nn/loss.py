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
    from ml_switcheroo_compiler.ops.binary import multiply, add, subtract, true_divide
    from ml_switcheroo_compiler.ops.reductions import sum as op_sum

    intersection = op_sum(multiply(y_true, y_pred), axis=axis)
    y_true_sum = op_sum(y_true, axis=axis)
    y_pred_sum = op_sum(y_pred, axis=axis)

    numerator = add(multiply(2.0, intersection), smooth)
    denominator = add(add(y_true_sum, y_pred_sum), smooth)

    dice_coeff = true_divide(numerator, denominator)
    return subtract(1.0, dice_coeff)


__all__ = ["ctc_loss", "dice_loss"]
