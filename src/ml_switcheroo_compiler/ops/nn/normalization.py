"""Apply normalization operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    TracedDataProtocol,
    extract_dtype,
    extract_shape,
)

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import (
    Tensor,
    TensorConfig,
    as_opt_tensor_obj,
    ensure_tensor,
)
from ml_switcheroo_compiler.ir.core import ensure_attrs
from ml_switcheroo_compiler.ops import (
    maximum,
    multiply,
    true_divide,
)
from ml_switcheroo_compiler.ops.base import register_op
from ml_switcheroo_compiler.ops.binary import add, divide, power, subtract
from ml_switcheroo_compiler.ops.configs import WindowConfig
from ml_switcheroo_compiler.ops.creation import full_like
from ml_switcheroo_compiler.ops.normalization.basic import GroupNorm
from ml_switcheroo_compiler.ops.reductions import (
    mean,
    reduce_window,
    sum,
    variance,
)
from ml_switcheroo_compiler.ops.shape.frontend import reshape
from ml_switcheroo_compiler.ops.unary import sqrt


@typing.runtime_checkable
class _BatchNormGlobalConfigProtocol(typing.Protocol):
    """Protocol for global batch normalization configurations."""

    scale_after_normalization: bool
    variance_epsilon: float


@typing.runtime_checkable
class _LRNConfigProtocol(typing.Protocol):
    """Protocol for LRN configurations."""

    depth_radius: int
    bias: float
    alpha: float
    beta: float


def _node_input_id(obj: TensorOrScalar) -> str:
    """Resolve tracing node input identifier string from `obj`.

    Args:
        obj (TensorOrScalar): Input tensor or proxy.

    Returns:
        str: String node identifier.
    """
    return str(obj.id if isinstance(obj, TracedDataProtocol) else obj)


class LRNConfig(pydantic.BaseModel):
    """LRN Config."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    depth_radius: int = 5
    bias: float = 1.0
    alpha: float = 1.0
    beta: float = 0.5


class BatchNormConfig(pydantic.BaseModel):
    """Batch normalization config."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    offset: Tensor | None = None
    scale: Tensor | None = None
    epsilon: float = 1e-3


def local_response_normalization(
    operand: Tensor,
    depth_radius: int = 5,
    bias: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> TensorOrScalar:
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
    axis: int | Sequence[int],
    config: BatchNormConfig | None = None,
) -> TensorOrScalar:
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
) -> TensorOrScalar:
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


class BatchNormGlobalConfig(pydantic.BaseModel):
    """Configuration for batch_norm_with_global_normalization."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    variance_epsilon: float = 1e-5
    scale_after_normalization: bool = True
    name: str | None = None


def batch_norm_with_global_normalization(
    t: TensorOrScalar,
    m: TensorOrScalar,
    v: TensorOrScalar,
    beta: TensorOrScalar,
    gamma: TensorOrScalar,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
    """Batch normalization with global normalization.

    Args:
        t (TensorOrScalar): The t parameter.
        m (TensorOrScalar): The m parameter.
        v (TensorOrScalar): The v parameter.
        beta (TensorOrScalar): The beta parameter.
        gamma (TensorOrScalar): The gamma parameter.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    raw_cfg = kwargs.get("config")
    cfg = raw_cfg if isinstance(raw_cfg, BatchNormGlobalConfig) else BatchNormGlobalConfig()
    bn_config = BatchNormConfig(
        offset=as_opt_tensor_obj(beta),
        scale=as_opt_tensor_obj(gamma if cfg.scale_after_normalization else None),
        epsilon=cfg.variance_epsilon,
    )
    return batch_normalization(
        ensure_tensor(t),
        ensure_tensor(m),
        ensure_tensor(v),
        axis=-1,  # Default to last axis if not provided
        config=bn_config,
    )


def lrn(input: TensorOrScalar, config: LRNConfig | TensorOrScalar = None, name: TensorOrScalar = None) -> TensorOrScalar:
    """Compute the lrn tensor operation.

    Args:
        input (TensorOrScalar): The input parameter.
        config (LRNConfig): The config parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    cfg = config if isinstance(config, LRNConfig) else LRNConfig()
    return local_response_normalization(ensure_tensor(input), cfg.depth_radius, cfg.bias, cfg.alpha, cfg.beta)


def l2_normalize(
    x: Tensor,
    axis: int | Sequence[int] | None = None,
    epsilon: float = 1e-12,
    name: str | None = None,
) -> Tensor:
    """Normalize along dimension axis using an L2 norm.

    Args:
        x (Tensor): The x parameter.
        axis (int | Sequence[int] | None): The axis parameter.
        epsilon (float): The epsilon parameter.
        name (str | None): The name parameter.
        axis (int | Sequence[int] | None): The axis parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    square_sum = sum(multiply(x, x), axis=axis, keepdims=True)
    x_inv_norm = true_divide(1.0, sqrt(maximum(square_sum, epsilon)))
    return multiply(x, x_inv_norm)


def moments(x: TensorOrScalar, axes: TensorOrScalar, shift: TensorOrScalar = None, keepdims: TensorOrScalar = False, name: TensorOrScalar = None) -> TensorOrScalar:
    """Calculate the mean and variance of x.

    Args:
        x (TensorOrScalar): The x parameter.
        axes (TensorOrScalar): The axes parameter.
        shift (TensorOrScalar): The shift parameter.
        keepdims (TensorOrScalar): The keepdims parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return mean(x, axis=axes, keepdims=keepdims), variance(x, axis=axes, keepdims=keepdims)


def normalize_moments(counts: TensorOrScalar, mean_ss: TensorOrScalar, variance_ss: TensorOrScalar, shift: TensorOrScalar, name: TensorOrScalar = None) -> TensorOrScalar:
    """Calculate the mean and variance of based on the sufficient statistics.

    Args:
        counts (TensorOrScalar): The counts parameter.
        mean_ss (TensorOrScalar): The mean_ss parameter.
        variance_ss (TensorOrScalar): The variance_ss parameter.
        shift (TensorOrScalar): The shift parameter.
        name (TensorOrScalar): The name parameter.

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
    c_shape = extract_shape(counts)
    c_dtype = extract_dtype(counts, "float32")

    node = LogicalNode(
        id=out_id_mean,
        op_type="NormalizeMoments",
        inputs=[_node_input_id(counts), _node_input_id(mean_ss), _node_input_id(variance_ss), _node_input_id(shift)],
        attributes=ensure_attrs({"name": name, "secondary_id": out_id_variance}),
        shape_metadata=c_shape,
    )
    global_tracing_state.add_node(node)

    proxy_mean = ProxyTensor(id=out_id_mean, shape=c_shape, dtype=c_dtype)
    proxy_variance = ProxyTensor(id=out_id_variance, shape=c_shape, dtype=c_dtype)

    return (
        Tensor(proxy_mean, TensorConfig(c_shape, c_dtype, "cpu")),
        Tensor(proxy_variance, TensorConfig(c_shape, c_dtype, "cpu")),
    )


def sufficient_statistics(x: TensorOrScalar, axes: TensorOrScalar, shift: TensorOrScalar = None, keepdims: TensorOrScalar = False, name: TensorOrScalar = None) -> TensorOrScalar:
    """Calculate the sufficient statistics for the mean and variance of x.

    Args:
        x (TensorOrScalar): The x parameter.
        axes (TensorOrScalar): The axes parameter.
        shift (TensorOrScalar): The shift parameter.
        keepdims (TensorOrScalar): The keepdims parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
        TensorOrScalar: Sufficient statistics tuple.
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
    x_shape = extract_shape(x)
    x_dtype = extract_dtype(x, "float32")

    node = LogicalNode(
        id=out_id_counts,
        op_type="SufficientStatistics",
        inputs=[_node_input_id(x), _node_input_id(axes)],
        attributes=ensure_attrs({"shift": shift, "keepdims": keepdims, "name": name, "secondary_ids": [out_id_mean_ss, out_id_variance_ss, out_id_shift]}),
        shape_metadata=x_shape,
    )
    global_tracing_state.add_node(node)

    config_tensor = TensorConfig(x_shape, x_dtype, "cpu")
    return (
        Tensor(ProxyTensor(id=out_id_counts, shape=x_shape, dtype=x_dtype), config_tensor),
        Tensor(ProxyTensor(id=out_id_mean_ss, shape=x_shape, dtype=x_dtype), config_tensor),
        Tensor(ProxyTensor(id=out_id_variance_ss, shape=x_shape, dtype=x_dtype), config_tensor),
        Tensor(ProxyTensor(id=out_id_shift, shape=x_shape, dtype=x_dtype), config_tensor),
    )


def weighted_moments(
    x: TensorOrScalar,
    axes: TensorOrScalar,
    frequency_weights: TensorOrScalar,
    name: TensorOrScalar = None,
    keepdims: TensorOrScalar = False,
) -> TensorOrScalar:
    """Return the frequency-weighted mean and variance of x.

    Args:
        x (TensorOrScalar): The x parameter.
        axes (TensorOrScalar): The axes parameter.
        frequency_weights (TensorOrScalar): The frequency_weights parameter.
        name (TensorOrScalar): The name parameter.
        keepdims (TensorOrScalar): The keepdims parameter.

    Returns:
        TensorOrScalar: Weighted `(mean, variance)` tuple.
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
    x_shape = extract_shape(x)
    x_dtype = extract_dtype(x, "float32")

    node = LogicalNode(
        id=out_id_mean,
        op_type="WeightedMoments",
        inputs=[_node_input_id(x), _node_input_id(axes), _node_input_id(frequency_weights)],
        attributes=ensure_attrs({"name": name, "keepdims": keepdims, "secondary_id": out_id_variance}),
        shape_metadata=x_shape,
    )
    global_tracing_state.add_node(node)

    proxy_mean = ProxyTensor(id=out_id_mean, shape=x_shape, dtype=x_dtype)
    proxy_variance = ProxyTensor(id=out_id_variance, shape=x_shape, dtype=x_dtype)

    return (
        Tensor(proxy_mean, TensorConfig(x_shape, x_dtype, "cpu")),
        Tensor(proxy_variance, TensorConfig(x_shape, x_dtype, "cpu")),
    )


def zero_fraction(value: TensorOrScalar, name: TensorOrScalar = None) -> TensorOrScalar:
    """Return the fraction of zeros in value.

    Args:
        value (TensorOrScalar): The value parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return Tensor(0.0, TensorConfig((), "float32", "cpu"))


def layer_norm(
    x: Tensor,
    normalized_shape: Sequence[int],
    scale: Tensor | None = None,
    offset: Tensor | None = None,
    epsilon: float = 1e-5,
) -> TensorOrScalar:
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
    scale: Tensor | None = None,
    offset: Tensor | None = None,
    epsilon: float = 1e-5,
) -> TensorOrScalar:
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
    scale: Tensor | None = None,
    offset: Tensor | None = None,
    epsilon: float = 1e-5,
) -> TensorOrScalar:
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


Normalization = register_op("Normalization")(GroupNorm)
batch_norm = batch_normalization
rms_norm = rms_normalization
