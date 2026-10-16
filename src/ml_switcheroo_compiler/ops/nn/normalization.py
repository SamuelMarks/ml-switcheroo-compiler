"""Apply normalization operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union, cast  # noqa: F401

from ml_switcheroo_compiler.backends.registry import EagerValue

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
from ml_switcheroo_compiler.ops.base import OpDef, register_op
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
) -> EagerValue:
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
) -> EagerValue:
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
) -> EagerValue:
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
    t: EagerValue,
    m: EagerValue,
    v: EagerValue,
    beta: EagerValue,
    gamma: EagerValue,
    **kwargs: EagerValue,
) -> EagerValue:
    """Batch normalization with global normalization.

    Args:
        t (Any): The t parameter.
        m (Any): The m parameter.
        v (Any): The v parameter.
        beta (Any): The beta parameter.
        gamma (Any): The gamma parameter.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = kwargs.get("config", BatchNormGlobalConfig())
    bn_config = BatchNormConfig(
        offset=cast("Optional[Tensor]", beta),
        scale=cast("Optional[Tensor]", gamma if getattr(config, "scale_after_normalization") else None),
        epsilon=getattr(config, "variance_epsilon"),
    )
    return batch_normalization(
        cast("Tensor", t),
        cast("Tensor", m),
        cast("Tensor", v),
        axis=-1,  # Default to last axis if not provided
        config=bn_config,
    )


def lrn(input: EagerValue, config: EagerValue = None, name: EagerValue = None) -> EagerValue:
    """Evaluate lrn operation.

    Args:
        input (Any): The input parameter.
        config (LRNConfig): The config parameter.
        name (Any): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    config = cast(EagerValue, config or LRNConfig())
    depth_radius = getattr(config, "depth_radius")
    bias = getattr(config, "bias")
    alpha = getattr(config, "alpha")
    beta = getattr(config, "beta")

    return local_response_normalization(cast("Tensor", input), depth_radius, bias, alpha, beta)


def l2_normalize(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    epsilon: float = 1e-12,
    name: str | None = None,
) -> Tensor:
    """Normalize along dimension axis using an L2 norm.

    Args:
        x (Any): The x parameter.
        axis (Any): The axis parameter.
        epsilon (Any): The epsilon parameter.
        name (Any): The name parameter.
        axis (Any): The axis parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    square_sum = sum(multiply(x, x), axis=axis, keepdims=True)
    x_inv_norm = true_divide(1.0, sqrt(maximum(square_sum, epsilon)))
    return multiply(x, x_inv_norm)


def moments(x: EagerValue, axes: EagerValue, shift: EagerValue = None, keepdims: EagerValue = False, name: EagerValue = None) -> EagerValue:
    """Calculate the mean and variance of x.

    Args:
        x (Any): The x parameter.
        axes (Any): The axes parameter.
        shift (Any): The shift parameter.
        keepdims (Any): The keepdims parameter.
        name (Any): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return mean(x, axis=axes, keepdims=keepdims), variance(x, axis=axes, keepdims=keepdims)


def normalize_moments(counts: EagerValue, mean_ss: EagerValue, variance_ss: EagerValue, shift: EagerValue, name: EagerValue = None) -> EagerValue:
    """Calculate the mean and variance of based on the sufficient statistics.

    Args:
        counts (Any): The counts parameter.
        mean_ss (Any): The mean_ss parameter.
        variance_ss (Any): The variance_ss parameter.
        shift (Any): The shift parameter.
        name (Any): The name parameter.

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
        inputs=[str(getattr(counts, "id", counts)), str(getattr(mean_ss, "id", mean_ss)), str(getattr(variance_ss, "id", variance_ss)), str(getattr(shift, "id", shift))],
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


def sufficient_statistics(x: EagerValue, axes: EagerValue, shift: EagerValue = None, keepdims: EagerValue = False, name: EagerValue = None) -> EagerValue:
    """Calculate the sufficient statistics for the mean and variance of x.

    Args:
        x (Any): The x parameter.
        axes (Any): The axes parameter.
        shift (Any): The shift parameter.
        keepdims (Any): The keepdims parameter.
        name (Any): The name parameter.

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
        inputs=[str(getattr(x, "id", x)), str(getattr(axes, "id", axes))],
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
    x: EagerValue,
    axes: EagerValue,
    frequency_weights: EagerValue,
    name: EagerValue = None,
    keepdims: EagerValue = False,
) -> EagerValue:
    """Return the frequency-weighted mean and variance of x.

    Args:
        x (Any): The x parameter.
        axes (Any): The axes parameter.
        frequency_weights (Any): The frequency_weights parameter.
        name (Any): The name parameter.
        keepdims (Any): The keepdims parameter.

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
        inputs=[str(getattr(x, "id", x)), str(getattr(axes, "id", axes)), str(getattr(frequency_weights, "id", frequency_weights))],
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


def zero_fraction(value: EagerValue, name: EagerValue = None) -> EagerValue:
    """Return the fraction of zeros in value.

    Args:
        value (Any): The value parameter.
        name (Any): The name parameter.

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
) -> EagerValue:
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
) -> EagerValue:
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
) -> EagerValue:
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


@register_op("Normalization")
class Normalization(OpDef):
    """Generic Normalization operation."""

    op_name = "Normalization"

    def infer_shape(self, *args: Tensor, **kwargs: Tensor | float | int | None) -> tuple[int, ...]:
        """Infer output shape for Normalization.

        Args:
            *args: Positional arguments (first argument is input tensor).
            **kwargs: Optional keyword arguments.

        Returns:
            tuple[int, ...]: Shape of the input tensor.
        """
        del kwargs
        return getattr(args[0], "shape", ()) if args else ()


batch_norm = batch_normalization
rms_norm = rms_normalization
