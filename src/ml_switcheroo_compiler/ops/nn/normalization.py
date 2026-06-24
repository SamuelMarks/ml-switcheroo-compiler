"""Normalization operations."""

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.binary import add, divide, multiply, power
from ml_switcheroo_compiler.ops.creation import full_like
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.reductions import reduce_window
from ml_switcheroo_compiler.ops.binary import subtract
from typing import Union, Optional
from collections.abc import Sequence
from ml_switcheroo_compiler.ops.reductions import mean

from dataclasses import dataclass


@dataclass
class BatchNormConfig:
    """Batch normalization config."""

    offset: Optional[Tensor] = None
    scale: Optional[Tensor] = None
    epsilon: float = 1e-3


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
    squared = multiply(operand, operand)  # pragma: no cover

    window_size = 2 * depth_radius + 1  # pragma: no cover
    rank = len(operand.shape)  # pragma: no cover

    window_dimensions = (1,) * (rank - 1) + (window_size,)  # pragma: no cover
    window_strides = (1,) * rank  # pragma: no cover

    padding = [(0, 0)] * (rank - 1) + [(depth_radius, depth_radius)]  # pragma: no cover

    config = WindowConfig(  # pragma: no cover
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding,
    )

    sqr_sum = reduce_window(squared, 0.0, "sum", config)  # pragma: no cover

    b_tensor = full_like(operand, bias)  # pragma: no cover
    a_tensor = full_like(operand, alpha)  # pragma: no cover
    beta_tensor = full_like(operand, beta)  # pragma: no cover

    scaled_sqr_sum = multiply(sqr_sum, a_tensor)  # pragma: no cover
    denom = add(b_tensor, scaled_sqr_sum)  # pragma: no cover
    denom_beta = power(denom, beta_tensor)  # pragma: no cover

    return divide(operand, denom_beta)  # pragma: no cover


def batch_normalization(
    x: Tensor,
    mean: Tensor,
    variance: Tensor,
    axis: Union[int, Sequence[int]],
    config: Optional[BatchNormConfig] = None,
) -> Tensor:
    """Batch normalization.

    Args:
        x (Tensor): Input tensor.
        mean (Tensor): Mean tensor.
        variance (Tensor): Variance tensor.
        axis (Union[int, Sequence[int]]): Axis that should be normalized.
        config (Optional[BatchNormConfig]): Config.

    Returns:
        Tensor: Normalized tensor.
    """
    conf = config if config is not None else BatchNormConfig()
    eps_tensor = full_like(variance, conf.epsilon)
    var_plus_eps = add(variance, eps_tensor)
    half = full_like(var_plus_eps, 0.5)
    stddev = power(var_plus_eps, half)

    x_minus_mean = subtract(x, mean)
    normalized_x = divide(x_minus_mean, stddev)

    if conf.scale is not None:  # pragma: no branch
        normalized_x = multiply(normalized_x, conf.scale)
    if conf.offset is not None:  # pragma: no branch
        normalized_x = add(normalized_x, conf.offset)

    return normalized_x


def rms_normalization(
    x: Tensor,
    scale: Tensor,
    epsilon: float = 1e-3,
) -> Tensor:
    """RMS normalization.

    Args:
        x (Tensor): Input tensor.
        scale (Tensor): Scale tensor.
        epsilon (float): Epsilon.

    Returns:
        Tensor: Normalized tensor.
    """
    squared_x = multiply(x, x)
    mean_sqr = mean(squared_x, axis=-1, keepdims=True)
    eps_tensor = full_like(mean_sqr, epsilon)
    mean_sqr_plus_eps = add(mean_sqr, eps_tensor)
    half = full_like(mean_sqr_plus_eps, 0.5)
    rms = power(mean_sqr_plus_eps, half)
    normalized = divide(x, rms)
    return multiply(normalized, scale)
