"""Normalization operations."""

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, divide, multiply, power
from ml_switcheroo_compiler.ops.creation import full_like
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.reductions import reduce_window


def local_response_normalization(
    operand: Tensor,
    depth_radius: int = 5,
    bias: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> Tensor:
    """Local Response Normalization.

    Args:
        operand (Tensor): The input tensor (batch, ..., channels).
        depth_radius (int): The radius of the half-window.
        bias (float): An offset.
        alpha (float): A scale factor.
        beta (float): An exponent.

    Returns:
        Tensor: The normalized tensor.
    """
    squared = multiply(operand, operand)

    window_size = 2 * depth_radius + 1
    rank = len(operand.shape)

    window_dimensions = (1,) * (rank - 1) + (window_size,)
    window_strides = (1,) * rank

    padding = [(0, 0)] * (rank - 1) + [(depth_radius, depth_radius)]

    config = WindowConfig(
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding,
    )

    sqr_sum = reduce_window(squared, 0.0, "sum", config)

    b_tensor = full_like(operand, bias)
    a_tensor = full_like(operand, alpha)
    beta_tensor = full_like(operand, beta)

    scaled_sqr_sum = multiply(sqr_sum, a_tensor)
    denom = add(b_tensor, scaled_sqr_sum)
    denom_beta = power(denom, beta_tensor)

    return divide(operand, denom_beta)
