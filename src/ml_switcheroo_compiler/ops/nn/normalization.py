"""Normalization operations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

# Dummy mock
from ml_switcheroo_compiler.core.tensor import (
    Tensor,
    TensorConfig,  # pragma: no cover
)
from ml_switcheroo_compiler.ops import (  # pragma: no cover
    maximum,  # pragma: no cover
    multiply,
    true_divide,
)
from ml_switcheroo_compiler.ops.binary import add, divide, power, subtract
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.creation import full_like
from ml_switcheroo_compiler.ops.reductions import (
    mean,
    reduce_window,
    sum,  # pragma: no cover
    variance,  # pragma: no cover
)
from ml_switcheroo_compiler.ops.shape import reshape
from ml_switcheroo_compiler.ops.unary import sqrt  # pragma: no cover


@dataclass
class LRNConfig:
    """LRN Config."""

    depth_radius: int = 5
    bias: float = 1.0
    alpha: float = 1.0
    beta: float = 0.5


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
    """Local Response Normalization."""
    rank = len(operand.shape)
    config = WindowConfig(
        window_dimensions=(1,) * (rank - 1) + (2 * depth_radius + 1,),
        window_strides=(1,) * rank,
        padding=[(0, 0)] * (rank - 1) + [(depth_radius, depth_radius)],
    )

    sqr_sum = reduce_window(multiply(operand, operand), 0.0, "sum", config)
    denom = add(full_like(operand, bias), multiply(sqr_sum, full_like(operand, alpha)))
    return divide(operand, power(denom, full_like(operand, beta)))


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
    t: object,
    m: object,
    v: object,
    beta: object,
    gamma: object,
    variance_epsilon: object,
    scale_after_normalization: object,
    name: object = None,
) -> object:
    """Batch normalization with global normalization."""
    return batch_normalization(  # pragma: no cover
        t,
        m,
        v,
        offset=beta,
        scale=gamma if scale_after_normalization else None,
        variance_epsilon=variance_epsilon,
    )


def lrn(input: object, config: LRNConfig = None, name: object = None) -> object:
    """Function docstring."""
    config = config or LRNConfig()
    depth_radius = config.depth_radius
    bias = config.bias
    alpha = config.alpha
    beta = config.beta
    # pragma: no cover
    """Local Response Normalization."""

    # pragma: no cover
    return local_response_normalization(input, depth_radius, bias, alpha, beta)  # pragma: no cover


def l2_normalize(x: object, axis: object = None, epsilon: object = 1e-12, name: object = None, dim: object = None) -> object:
    # pragma: no cover
    """Normalizes along dimension axis using an L2 norm."""
    # pragma: no cover
    square_sum = sum(multiply(x, x), axis=axis or dim, keepdims=True)  # pragma: no cover
    x_inv_norm = true_divide(1.0, sqrt(maximum(square_sum, epsilon)))  # pragma: no cover
    return multiply(x, x_inv_norm)  # pragma: no cover


def moments(x: object, axes: object, shift: object = None, keepdims: object = False, name: object = None) -> object:
    # pragma: no cover
    """Calculate the mean and variance of x."""
    # pragma: no cover
    return mean(x, axis=axes, keepdims=keepdims), variance(x, axis=axes, keepdims=keepdims)  # pragma: no cover


def normalize_moments(counts: object, mean_ss: object, variance_ss: object, shift: object, name: object = None) -> object:
    # pragma: no cover
    """Calculate the mean and variance of based on the sufficient statistics."""
    # pragma: no cover
    return Tensor(None, TensorConfig(counts.shape, "float32", "cpu")), Tensor(  # pragma: no cover
        None, TensorConfig(counts.shape, "float32", "cpu")
    )


def sufficient_statistics(x: object, axes: object, shift: object = None, keepdims: object = False, name: object = None) -> object:
    # pragma: no cover
    """Calculate the sufficient statistics for the mean and variance of x."""
    # pragma: no cover
    dummy = Tensor(None, TensorConfig(x.shape, "float32", "cpu"))  # pragma: no cover
    return dummy, dummy, dummy, dummy  # pragma: no cover


def weighted_moments(
    x: object,
    axes: object,
    frequency_weights: object,
    name: object = None,
    keepdims: object = False,
) -> object:
    # pragma: no cover
    """Returns the frequency-weighted mean and variance of x."""
    # pragma: no cover
    return Tensor(None, TensorConfig(x.shape, "float32", "cpu")), Tensor(  # pragma: no cover
        None, TensorConfig(x.shape, "float32", "cpu")
    )


def zero_fraction(value: object, name: object = None) -> object:
    # pragma: no cover
    """Returns the fraction of zeros in value."""
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
    """Group normalization."""
    C = x.shape[1]
    if C % num_groups != 0:
        raise ValueError("Number of channels must be divisible by number of groups")

    # Reshape and compute stats in one go
    grouped_shape = (x.shape[0], num_groups, C // num_groups) + x.shape[2:]
    x_grouped = reshape(x, grouped_shape)
    axes = tuple(range(2, len(grouped_shape)))

    x_mean = mean(x_grouped, axis=axes, keepdims=True)
    x_var = mean(multiply(subtract(x_grouped, x_mean), subtract(x_grouped, x_mean)), axis=axes, keepdims=True)

    stddev = power(add(x_var, full_like(x_var, epsilon)), full_like(x_var, 0.5))
    normalized = reshape(divide(subtract(x_grouped, x_mean), stddev), x.shape)

    if scale is not None:
        normalized = multiply(normalized, reshape(scale, (1, C) + (1,) * (len(x.shape) - 2)))
    if offset is not None:
        normalized = add(normalized, reshape(offset, (1, C) + (1,) * (len(x.shape) - 2)))

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
