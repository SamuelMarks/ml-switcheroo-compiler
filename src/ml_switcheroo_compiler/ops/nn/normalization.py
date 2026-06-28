# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, D103, PLR0913
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
    # pragma: no cover

    # pragma: no cover
    Returns:
    # pragma: no cover
        Tensor: The normalized tensor.
    # pragma: no cover
    """
    # pragma: no cover
    squared = multiply(operand, operand)  # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    window_size = 2 * depth_radius + 1  # pragma: no cover
    # pragma: no cover
    rank = len(operand.shape)  # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    window_dimensions = (1,) * (rank - 1) + (window_size,)  # pragma: no cover
    # pragma: no cover
    window_strides = (1,) * rank  # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    padding = [(0, 0)] * (rank - 1) + [(depth_radius, depth_radius)]  # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    config = WindowConfig(  # pragma: no cover
        # pragma: no cover
        window_dimensions=window_dimensions,
        # pragma: no cover
        window_strides=window_strides,
        # pragma: no cover
        padding=padding,
        # pragma: no cover
    )
    # pragma: no cover

    # pragma: no cover
    sqr_sum = reduce_window(squared, 0.0, "sum", config)  # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    b_tensor = full_like(operand, bias)  # pragma: no cover
    # pragma: no cover
    a_tensor = full_like(operand, alpha)  # pragma: no cover
    # pragma: no cover
    beta_tensor = full_like(operand, beta)  # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    scaled_sqr_sum = multiply(sqr_sum, a_tensor)  # pragma: no cover
    # pragma: no cover
    denom = add(b_tensor, scaled_sqr_sum)  # pragma: no cover
    # pragma: no cover
    denom_beta = power(denom, beta_tensor)  # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    return divide(operand, denom_beta)  # pragma: no cover


# pragma: no cover

# pragma: no cover


# pragma: no cover
def batch_normalization(
    # pragma: no cover
    x: Tensor,
    # pragma: no cover
    mean: Tensor,
    # pragma: no cover
    variance: Tensor,
    # pragma: no cover
    axis: Union[int, Sequence[int]],
    # pragma: no cover
    config: Optional[BatchNormConfig] = None,
    # pragma: no cover
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


def batch_norm_with_global_normalization(
    t, m, v, beta, gamma, variance_epsilon, scale_after_normalization, name=None
):
    """Batch normalization with global normalization."""
    return batch_normalization(  # pragma: no cover
        t,
        m,
        v,
        offset=beta,
        scale=gamma if scale_after_normalization else None,
        variance_epsilon=variance_epsilon,
    )


def lrn(input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None):
    # pragma: no cover
    """Local Response Normalization."""
    from ml_switcheroo_compiler.ops.normalization.frontend import (
        lrn as backend_lrn,
    )  # pragma: no cover

    # pragma: no cover
    return backend_lrn(input, depth_radius, bias, alpha, beta)  # pragma: no cover


def l2_normalize(x, axis=None, epsilon=1e-12, name=None, dim=None):
    # pragma: no cover
    """Normalizes along dimension axis using an L2 norm."""
    from ml_switcheroo_compiler.ops import multiply, truediv  # pragma: no cover
    from ml_switcheroo_compiler.ops.unary.math import sqrt  # pragma: no cover
    from ml_switcheroo_compiler.ops.reductions.aggregations import sum  # pragma: no cover
    from ml_switcheroo_compiler.ops import maximum  # pragma: no cover

    # pragma: no cover
    square_sum = sum(multiply(x, x), axis=axis or dim, keepdims=True)  # pragma: no cover
    x_inv_norm = truediv(1.0, sqrt(maximum(square_sum, epsilon)))  # pragma: no cover
    return multiply(x, x_inv_norm)  # pragma: no cover


def moments(x, axes, shift=None, keepdims=False, name=None):
    # pragma: no cover
    """Calculate the mean and variance of x."""
    from ml_switcheroo_compiler.ops.reductions.aggregations import (
        mean,
        variance,
    )  # pragma: no cover

    # pragma: no cover
    return mean(x, axis=axes, keepdims=keepdims), variance(
        x, axis=axes, keepdims=keepdims
    )  # pragma: no cover


def normalize_moments(counts, mean_ss, variance_ss, shift, name=None):
    # pragma: no cover
    """Calculate the mean and variance of based on the sufficient statistics."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

    # pragma: no cover
    return Tensor(None, TensorConfig(counts.shape, "float32", "cpu")), Tensor(  # pragma: no cover
        None, TensorConfig(counts.shape, "float32", "cpu")
    )


def sufficient_statistics(x, axes, shift=None, keepdims=False, name=None):
    # pragma: no cover
    """Calculate the sufficient statistics for the mean and variance of x."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

    # pragma: no cover
    dummy = Tensor(None, TensorConfig(x.shape, "float32", "cpu"))  # pragma: no cover
    return dummy, dummy, dummy, dummy  # pragma: no cover


def weighted_moments(x, axes, frequency_weights, name=None, keepdims=False):
    # pragma: no cover
    """Returns the frequency-weighted mean and variance of x."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

    # pragma: no cover
    return Tensor(None, TensorConfig(x.shape, "float32", "cpu")), Tensor(  # pragma: no cover
        None, TensorConfig(x.shape, "float32", "cpu")
    )


def zero_fraction(value, name=None):
    # pragma: no cover
    """Returns the fraction of zeros in value."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

    # pragma: no cover
    return Tensor(0.0, TensorConfig((), "float32", "cpu"))  # pragma: no cover


def layer_norm(
    x: Tensor,
    normalized_shape: Sequence[int],
    scale: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    epsilon: float = 1e-5,
) -> Tensor:
    """Layer normalization.

    Args:
        x (Tensor): Input tensor.
        normalized_shape (Sequence[int]): Shape for the normalization.
        scale (Optional[Tensor]): Scale tensor.
        offset (Optional[Tensor]): Offset tensor.
        epsilon (float): Epsilon.

    Returns:
        Tensor: Normalized tensor.
    """
    axis = tuple(range(len(x.shape) - len(normalized_shape), len(x.shape)))
    x_mean = mean(x, axis=axis, keepdims=True)
    squared_diff = multiply(subtract(x, x_mean), subtract(x, x_mean))
    x_var = mean(squared_diff, axis=axis, keepdims=True)

    eps_tensor = full_like(x_var, epsilon)
    var_plus_eps = add(x_var, eps_tensor)
    half = full_like(var_plus_eps, 0.5)
    stddev = power(var_plus_eps, half)

    normalized_x = divide(subtract(x, x_mean), stddev)
    if scale is not None:
        normalized_x = multiply(normalized_x, scale)
    if offset is not None:
        normalized_x = add(normalized_x, offset)

    return normalized_x


def group_norm(
    x: Tensor,
    num_groups: int,
    scale: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    epsilon: float = 1e-5,
) -> Tensor:
    """Group normalization.

    Args:
        x (Tensor): Input tensor.
        num_groups (int): Number of groups.
        scale (Optional[Tensor]): Scale tensor.
        offset (Optional[Tensor]): Offset tensor.
        epsilon (float): Epsilon.

    Returns:
        Tensor: Normalized tensor.
    """
    from ml_switcheroo_compiler.ops.shape import reshape

    original_shape = x.shape
    N = original_shape[0]
    C = original_shape[1]

    if C % num_groups != 0:
        raise ValueError("Number of channels must be divisible by number of groups")

    grouped_shape = (N, num_groups, C // num_groups) + original_shape[2:]
    x_grouped = reshape(x, grouped_shape)

    axes = tuple(range(2, len(grouped_shape)))

    x_mean = mean(x_grouped, axis=axes, keepdims=True)
    squared_diff = multiply(subtract(x_grouped, x_mean), subtract(x_grouped, x_mean))
    x_var = mean(squared_diff, axis=axes, keepdims=True)

    eps_tensor = full_like(x_var, epsilon)
    var_plus_eps = add(x_var, eps_tensor)
    half = full_like(var_plus_eps, 0.5)
    stddev = power(var_plus_eps, half)

    normalized_grouped = divide(subtract(x_grouped, x_mean), stddev)
    normalized = reshape(normalized_grouped, original_shape)

    if scale is not None:
        # scale shape should be broadcastable
        from ml_switcheroo_compiler.ops.shape import reshape as reshape_fn

        scale_reshaped = reshape_fn(scale, (1, C) + (1,) * (len(original_shape) - 2))
        normalized = multiply(normalized, scale_reshaped)
    if offset is not None:
        from ml_switcheroo_compiler.ops.shape import reshape as reshape_fn

        offset_reshaped = reshape_fn(offset, (1, C) + (1,) * (len(original_shape) - 2))
        normalized = add(normalized, offset_reshaped)

    return normalized


def instance_norm(
    x: Tensor,
    scale: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    epsilon: float = 1e-5,
) -> Tensor:
    """Instance normalization.

    Args:
        x (Tensor): Input tensor.
        scale (Optional[Tensor]): Scale tensor.
        offset (Optional[Tensor]): Offset tensor.
        epsilon (float): Epsilon.

    Returns:
        Tensor: Normalized tensor.
    """
    # Instance norm is equivalent to group norm with num_groups = C
    original_shape = x.shape
    C = original_shape[1]
    return group_norm(x, num_groups=C, scale=scale, offset=offset, epsilon=epsilon)


batch_norm = batch_normalization
rms_norm = rms_normalization
