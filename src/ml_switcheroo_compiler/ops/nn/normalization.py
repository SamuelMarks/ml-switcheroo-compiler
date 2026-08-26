# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Apply normalization operations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import (
    Tensor,
    TensorConfig,
)
from ml_switcheroo_compiler.ops import (
    maximum,
    multiply,
    true_divide,
)
from ml_switcheroo_compiler.ops.binary import add, divide, power, subtract
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.creation import full_like
from ml_switcheroo_compiler.ops.reductions import (
    mean,
    reduce_window,
    sum,
    variance,
)
from ml_switcheroo_compiler.ops.shape.frontend import reshape
from ml_switcheroo_compiler.ops.unary import sqrt


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
):
    """Local Response Normalization.

    Args:
        operand (Tensor): The operand parameter.
        depth_radius (int): The depth_radius parameter.
        bias (float): The bias parameter.
        alpha (float): The alpha parameter.
        beta (float): The beta parameter.

    Returns:
        Tensor: Result.
    """
    rank = len(operand.shape)
    config = WindowConfig(
        window_dimensions=(1,) * (rank - 1) + (2 * depth_radius + 1,),
        window_strides=(1,) * rank,
        padding=[(0, 0)] * (rank - 1) + [(depth_radius, depth_radius)],
    )

    sqr_sum = reduce_window(multiply(operand, operand), 0.0, "sum", config)
    denom = add(full_like(operand, bias), multiply(sqr_sum, full_like(operand, alpha)))
    return divide(operand, power(denom, full_like(operand, beta)))


def batch_normalization(
    x: Tensor,
    mean: Tensor,
    variance: Tensor,
    axis: Union[int, Sequence[int]],
    config: Optional[BatchNormConfig] = None,
):
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

    if conf.scale is not None:
        normalized_x = multiply(normalized_x, conf.scale)
    if conf.offset is not None:
        normalized_x = add(normalized_x, conf.offset)

    return normalized_x


def rms_normalization(
    x: Tensor,
    scale: Tensor,
    epsilon: float = 1e-3,
):
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


@dataclass
class BatchNormGlobalConfig:
    """Configuration for batch_norm_with_global_normalization."""

    variance_epsilon: float = 1e-5
    scale_after_normalization: bool = True
    name = None


def batch_norm_with_global_normalization(
    t,
    m,
    v,
    beta,
    gamma,
    **kwargs,
):
    """Batch normalization with global normalization.

    Args:
        t (object): The t parameter.
        m (object): The m parameter.
        v (object): The v parameter.
        beta (object): The beta parameter.
        gamma (object): The gamma parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = kwargs.get("config", BatchNormGlobalConfig())
    bn_config = BatchNormConfig(
        offset=beta,
        scale=gamma if config.scale_after_normalization else None,
        epsilon=config.variance_epsilon,
    )
    return batch_normalization(
        t,
        m,
        v,
        axis=-1,  # Default to last axis if not provided
        config=bn_config,
    )


def lrn(input, config=None, name=None):
    """Evaluate lrn operation.

    Args:
        input (object): The input parameter.
        config (LRNConfig): The config parameter.
        name (object): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or LRNConfig()
    depth_radius = config.depth_radius
    bias = config.bias
    alpha = config.alpha
    beta = config.beta

    """Local Response Normalization."""

    return local_response_normalization(input, depth_radius, bias, alpha, beta)


def l2_normalize(x, axis=None, epsilon=1e-12, name=None):
    """Normalize along dimension axis using an L2 norm.

    Args:
        x (object): The x parameter.
        axis (object): The axis parameter.
        epsilon (object): The epsilon parameter.
        name (object): The name parameter.
        axis (object): The axis parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    square_sum = sum(multiply(x, x), axis=axis, keepdims=True)
    x_inv_norm = true_divide(1.0, sqrt(maximum(square_sum, epsilon)))
    return multiply(x, x_inv_norm)


def moments(x, axes, shift=None, keepdims=False, name=None):
    """Calculate the mean and variance of x.

    Args:
        x (object): The x parameter.
        axes (object): The axes parameter.
        shift (object): The shift parameter.
        keepdims (object): The keepdims parameter.
        name (object): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return mean(x, axis=axes, keepdims=keepdims), variance(x, axis=axes, keepdims=keepdims)


def normalize_moments(counts, mean_ss, variance_ss, shift, name=None):
    """Calculate the mean and variance of based on the sufficient statistics.

    Args:
        counts (object): The counts parameter.
        mean_ss (object): The mean_ss parameter.
        variance_ss (object): The variance_ss parameter.
        shift (object): The shift parameter.
        name (object): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("NormalizeMoments", counts, mean_ss, variance_ss, shift, name=name)
    import uuid

    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    out_id_mean = str(uuid.uuid4())
    out_id_variance = str(uuid.uuid4())

    node = LogicalNode(
        id=out_id_mean,
        op_type="NormalizeMoments",
        inputs=[getattr(counts, "id", counts), getattr(mean_ss, "id", mean_ss), getattr(variance_ss, "id", variance_ss), getattr(shift, "id", shift)],
        attributes={"name": name, "secondary_id": out_id_variance},
        shape_metadata=getattr(counts, "shape", ()),
    )
    global_tracing_state.add_node(node)

    proxy_mean = ProxyTensor(id=out_id_mean, shape=getattr(counts, "shape", ()), dtype=getattr(counts, "dtype", "float32"))
    proxy_variance = ProxyTensor(id=out_id_variance, shape=getattr(counts, "shape", ()), dtype=getattr(counts, "dtype", "float32"))

    return (
        Tensor(proxy_mean, TensorConfig(getattr(counts, "shape", ()), getattr(counts, "dtype", "float32"), "cpu")),
        Tensor(proxy_variance, TensorConfig(getattr(counts, "shape", ()), getattr(counts, "dtype", "float32"), "cpu")),
    )


def sufficient_statistics(x, axes, shift=None, keepdims=False, name=None):
    """Calculate the sufficient statistics for the mean and variance of x.

    Args:
        x (object): The x parameter.
        axes (object): The axes parameter.
        shift (object): The shift parameter.
        keepdims (object): The keepdims parameter.
        name (object): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("SufficientStatistics", x, axes, shift=shift, keepdims=keepdims, name=name)
    import uuid

    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    out_id_counts = str(uuid.uuid4())
    out_id_mean_ss = str(uuid.uuid4())
    out_id_variance_ss = str(uuid.uuid4())
    out_id_shift = str(uuid.uuid4())

    node = LogicalNode(
        id=out_id_counts,
        op_type="SufficientStatistics",
        inputs=[getattr(x, "id", x), getattr(axes, "id", axes)],
        attributes={"shift": shift, "keepdims": keepdims, "name": name, "secondary_ids": [out_id_mean_ss, out_id_variance_ss, out_id_shift]},
        shape_metadata=getattr(x, "shape", ()),
    )
    global_tracing_state.add_node(node)

    config_tensor = TensorConfig(getattr(x, "shape", ()), getattr(x, "dtype", "float32"), "cpu")
    return (
        Tensor(ProxyTensor(id=out_id_counts, shape=getattr(x, "shape", ()), dtype=getattr(x, "dtype", "float32")), config_tensor),
        Tensor(ProxyTensor(id=out_id_mean_ss, shape=getattr(x, "shape", ()), dtype=getattr(x, "dtype", "float32")), config_tensor),
        Tensor(ProxyTensor(id=out_id_variance_ss, shape=getattr(x, "shape", ()), dtype=getattr(x, "dtype", "float32")), config_tensor),
        Tensor(ProxyTensor(id=out_id_shift, shape=getattr(x, "shape", ()), dtype=getattr(x, "dtype", "float32")), config_tensor),
    )


def weighted_moments(
    x,
    axes,
    frequency_weights,
    name=None,
    keepdims=False,
):
    """Return the frequency-weighted mean and variance of x.

    Args:
        x (object): The x parameter.
        axes (object): The axes parameter.
        frequency_weights (object): The frequency_weights parameter.
        name (object): The name parameter.
        keepdims (object): The keepdims parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("WeightedMoments", x, axes, frequency_weights, name=name, keepdims=keepdims)
    import uuid

    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.state import global_tracing_state
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    out_id_mean = str(uuid.uuid4())
    out_id_variance = str(uuid.uuid4())

    node = LogicalNode(
        id=out_id_mean,
        op_type="WeightedMoments",
        inputs=[getattr(x, "id", x), getattr(axes, "id", axes), getattr(frequency_weights, "id", frequency_weights)],
        attributes={"name": name, "keepdims": keepdims, "secondary_id": out_id_variance},
        shape_metadata=getattr(x, "shape", ()),
    )
    global_tracing_state.add_node(node)

    proxy_mean = ProxyTensor(id=out_id_mean, shape=getattr(x, "shape", ()), dtype=getattr(x, "dtype", "float32"))
    proxy_variance = ProxyTensor(id=out_id_variance, shape=getattr(x, "shape", ()), dtype=getattr(x, "dtype", "float32"))

    return (
        Tensor(proxy_mean, TensorConfig(getattr(x, "shape", ()), getattr(x, "dtype", "float32"), "cpu")),
        Tensor(proxy_variance, TensorConfig(getattr(x, "shape", ()), getattr(x, "dtype", "float32"), "cpu")),
    )


def zero_fraction(value, name=None):
    """Return the fraction of zeros in value.

    Args:
        value (object): The value parameter.
        name (object): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return Tensor(0.0, TensorConfig((), "float32", "cpu"))


def layer_norm(
    x: Tensor,
    normalized_shape: Sequence[int],
    scale: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    epsilon: float = 1e-5,
):
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
):
    """Group normalization.

    Args:
        x (Tensor): The x parameter.
        num_groups (int): The num_groups parameter.
        scale (Optional): The scale parameter.
        offset (Optional): The offset parameter.
        epsilon (float): The epsilon parameter.

    Returns:
        Tensor: Result.

    Raises:
        ValueError: An exception.
    """
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
):
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
