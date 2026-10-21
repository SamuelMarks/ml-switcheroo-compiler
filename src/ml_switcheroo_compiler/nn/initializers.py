"""Neural network parameter initializers."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable, Sequence

import ml_switcheroo_compiler.core.dtype as dtypes
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.backends.eager_registry import as_int
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.configs import InitializerConfig

InitializerFn = Callable[[typing.Union[Tensor, int, "None"], tuple[int, ...], typing.Union[dtypes.DType, str, "None"]], Tensor]


def _resolve_dtype(
    dtype: dtypes.DType | str | None,
    default: dtypes.DType | str = dtypes.DType.Float32,
) -> dtypes.DType:
    """Resolve an optional dtype specification into a canonical DType enum.

    Args:
        dtype (dtypes.DType | str | None): Optional requested data type.
        default (dtypes.DType | str): Default data type fallback.

    Returns:
        dtypes.DType: Resolved canonical DType enum.
    """
    target = dtype if dtype is not None else default
    return dtypes.DType(target)


def _compute_fans(
    shape: Sequence[int],
    in_axis: int = -2,
    out_axis: int = -1,
) -> tuple[int, int]:
    """Compute fan-in and fan-out dimensions for a weight tensor shape.

    Args:
        shape (Sequence[int]): Tensor shape sequence.
        in_axis (int): Input feature dimension axis.
        out_axis (int): Output feature dimension axis.

    Returns:
        tuple[int, int]: Tuple of (fan_in, fan_out).
    """
    ndim = len(shape)
    if ndim == 0:
        return 1, 1
    if ndim == 1:
        return shape[0], shape[0]
    norm_in = in_axis % ndim
    norm_out = out_axis % ndim
    receptive_field_size = 1
    for idx, dim in enumerate(shape):
        if idx not in (norm_in, norm_out):
            receptive_field_size *= dim
    fan_in = shape[norm_in] * receptive_field_size
    fan_out = shape[norm_out] * receptive_field_size
    return max(1, fan_in), max(1, fan_out)


def zeros(
    key: Tensor | int | None,
    shape: tuple[int, ...],
    dtype: dtypes.DType | str | None = dtypes.DType.Float32,
) -> Tensor:
    """Initialize a tensor filled with zeros.

    Args:
        key (Tensor | int | None): PRNG key or seed (unused for deterministic zeros).
        shape (tuple[int, ...]): Output tensor shape.
        dtype (dtypes.DType | str | None): Output tensor data type.

    Returns:
        Tensor: Zero-initialized tensor.
    """
    _ = key
    return ops.zeros(shape, dtype=_resolve_dtype(dtype))


def ones(
    key: Tensor | int | None,
    shape: tuple[int, ...],
    dtype: dtypes.DType | str | None = dtypes.DType.Float32,
) -> Tensor:
    """Initialize a tensor filled with ones.

    Args:
        key (Tensor | int | None): PRNG key or seed (unused for deterministic ones).
        shape (tuple[int, ...]): Output tensor shape.
        dtype (dtypes.DType | str | None): Output tensor data type.

    Returns:
        Tensor: One-initialized tensor.
    """
    _ = key
    return ops.ones(shape, dtype=_resolve_dtype(dtype))


def constant(
    value: float | int,
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build an initializer that fills a tensor with a constant scalar value.

    Args:
        value (float | int): Constant fill value.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: Initializer callable.
    """

    def init(
        key: Tensor | int | None,
        shape: tuple[int, ...],
        init_dtype: dtypes.DType | str | None = None,
    ) -> Tensor:
        """Initialize tensor with the configured constant value.

        Args:
            key (Tensor | int | None): PRNG key or seed.
            shape (tuple[int, ...]): Output tensor shape.
            init_dtype (dtypes.DType | str | None): Optional override data type.

        Returns:
            Tensor: Constant-initialized tensor.
        """
        _ = key
        return ops.full(shape, value, dtype=_resolve_dtype(init_dtype, dtype))

    return init


def uniform(
    scale: float = 1e-2,
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build an initializer sampling uniformly in [0, scale).

    Args:
        scale (float): Upper bound of the uniform interval [0, scale).
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: Initializer callable.
    """

    def init(
        key: Tensor | int | None,
        shape: tuple[int, ...],
        init_dtype: dtypes.DType | str | None = None,
    ) -> Tensor:
        """Sample uniform random values scaled by scale.

        Args:
            key (Tensor | int | None): PRNG key or seed.
            shape (tuple[int, ...]): Output tensor shape.
            init_dtype (dtypes.DType | str | None): Optional override data type.

        Returns:
            Tensor: Uniformly initialized tensor.
        """
        _ = key
        dt = _resolve_dtype(init_dtype, dtype)
        return ops.scalar_mul(ops.rand(*shape, dtype=dt), float(scale))

    return init


def normal(
    stddev: float = 1e-2,
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build an initializer sampling from a zero-mean normal distribution.

    Args:
        stddev (float): Standard deviation of the normal distribution.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: Initializer callable.
    """

    def init(
        key: Tensor | int | None,
        shape: tuple[int, ...],
        init_dtype: dtypes.DType | str | None = None,
    ) -> Tensor:
        """Sample normal random values scaled by stddev.

        Args:
            key (Tensor | int | None): PRNG key or seed.
            shape (tuple[int, ...]): Output tensor shape.
            init_dtype (dtypes.DType | str | None): Optional override data type.

        Returns:
            Tensor: Normally initialized tensor.
        """
        _ = key
        dt = _resolve_dtype(init_dtype, dtype)
        return ops.scalar_mul(ops.randn(*shape, dtype=dt), float(stddev))

    return init


def truncated_normal(
    stddev: float = 1e-2,
    dtype: dtypes.DType | str = dtypes.DType.Float32,
    lower: float = -2.0,
    upper: float = 2.0,
) -> InitializerFn:
    """Build an initializer sampling from a truncated normal distribution.

    Args:
        stddev (float): Standard deviation before or after truncation scaling.
        dtype (dtypes.DType | str): Default data type.
        lower (float): Lower truncation bound in standard deviations.
        upper (float): Upper truncation bound in standard deviations.

    Returns:
        InitializerFn: Initializer callable.
    """

    def init(
        key: Tensor | int | None,
        shape: tuple[int, ...],
        init_dtype: dtypes.DType | str | None = None,
    ) -> Tensor:
        """Sample truncated normal values within [lower * stddev, upper * stddev].

        Args:
            key (Tensor | int | None): PRNG key or seed.
            shape (tuple[int, ...]): Output tensor shape.
            init_dtype (dtypes.DType | str | None): Optional override data type.

        Returns:
            Tensor: Truncated-normal initialized tensor.
        """
        _ = key
        dt = _resolve_dtype(init_dtype, dtype)
        clipped = ops.clip(ops.randn(*shape, dtype=dt), float(lower), float(upper))
        return ops.scalar_mul(clipped, float(stddev))

    return init


def variance_scaling(config: InitializerConfig) -> InitializerFn:
    """Build a variance-scaling initializer adapting scale to weight fan-in/fan-out.

    Args:
        config (InitializerConfig): Variance scaling configuration parameters.

    Returns:
        InitializerFn: Initializer callable.
    """

    def init(
        key: Tensor | int | None,
        shape: tuple[int, ...],
        init_dtype: dtypes.DType | str | None = None,
    ) -> Tensor:
        """Initialize tensor according to variance scaling rules.

        Args:
            key (Tensor | int | None): PRNG key or seed.
            shape (tuple[int, ...]): Output tensor shape.
            init_dtype (dtypes.DType | str | None): Optional override data type.

        Returns:
            Tensor: Variance-scaled initialized tensor.
        """
        dt = typing.cast("TypedVariadicCallable[dtypes.DType]", _resolve_dtype)(init_dtype, config.dtype)
        fan_in, fan_out = _compute_fans(shape, as_int(config.in_axis), as_int(config.out_axis))
        if config.mode == "fan_in":
            denom = float(fan_in)
        elif config.mode == "fan_out":
            denom = float(fan_out)
        else:
            denom = (float(fan_in) + float(fan_out)) / 2.0
        variance = float(config.scale) / max(1.0, denom)

        if config.distribution == "truncated_normal":
            stddev = math.sqrt(variance) / 0.87962566103423978
            return truncated_normal(stddev=stddev, dtype=dt)(key, shape, dt)
        if config.distribution == "normal":
            stddev = math.sqrt(variance)
            return normal(stddev=stddev, dtype=dt)(key, shape, dt)
        limit = math.sqrt(3.0 * variance)
        unit = ops.rand(*shape, dtype=dt)
        centered = ops.subtract(ops.scalar_mul(unit, 2.0), ops.ones(shape, dtype=dt))
        return ops.scalar_mul(centered, limit)

    return init


def glorot_uniform(
    in_axis: int = -2,
    out_axis: int = -1,
    batch_axis: tuple[int, ...] = (),
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build a Glorot (Xavier) uniform initializer.

    Args:
        in_axis (int): Input dimension axis.
        out_axis (int): Output dimension axis.
        batch_axis (tuple[int, ...]): Batch axes.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: Glorot uniform initializer callable.
    """
    return variance_scaling(
        InitializerConfig(
            scale=1.0,
            mode="fan_avg",
            distribution="uniform",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=_resolve_dtype(dtype),
        )
    )


glorot = glorot_uniform
xavier_uniform = glorot_uniform


def glorot_normal(
    in_axis: int = -2,
    out_axis: int = -1,
    batch_axis: tuple[int, ...] = (),
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build a Glorot (Xavier) normal initializer.

    Args:
        in_axis (int): Input dimension axis.
        out_axis (int): Output dimension axis.
        batch_axis (tuple[int, ...]): Batch axes.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: Glorot normal initializer callable.
    """
    return variance_scaling(
        InitializerConfig(
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=_resolve_dtype(dtype),
        )
    )


xavier_normal = glorot_normal


def lecun_uniform(
    in_axis: int = -2,
    out_axis: int = -1,
    batch_axis: tuple[int, ...] = (),
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build a LeCun uniform initializer.

    Args:
        in_axis (int): Input dimension axis.
        out_axis (int): Output dimension axis.
        batch_axis (tuple[int, ...]): Batch axes.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: LeCun uniform initializer callable.
    """
    return variance_scaling(
        InitializerConfig(
            scale=1.0,
            mode="fan_in",
            distribution="uniform",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=_resolve_dtype(dtype),
        )
    )


def lecun_normal(
    in_axis: int = -2,
    out_axis: int = -1,
    batch_axis: tuple[int, ...] = (),
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build a LeCun normal initializer.

    Args:
        in_axis (int): Input dimension axis.
        out_axis (int): Output dimension axis.
        batch_axis (tuple[int, ...]): Batch axes.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: LeCun normal initializer callable.
    """
    return variance_scaling(
        InitializerConfig(
            scale=1.0,
            mode="fan_in",
            distribution="truncated_normal",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=_resolve_dtype(dtype),
        )
    )


def he_uniform(
    in_axis: int = -2,
    out_axis: int = -1,
    batch_axis: tuple[int, ...] = (),
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build a He (Kaiming) uniform initializer.

    Args:
        in_axis (int): Input dimension axis.
        out_axis (int): Output dimension axis.
        batch_axis (tuple[int, ...]): Batch axes.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: He uniform initializer callable.
    """
    return variance_scaling(
        InitializerConfig(
            scale=2.0,
            mode="fan_in",
            distribution="uniform",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=_resolve_dtype(dtype),
        )
    )


kaiming_uniform = he_uniform


def he_normal(
    in_axis: int = -2,
    out_axis: int = -1,
    batch_axis: tuple[int, ...] = (),
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build a He (Kaiming) normal initializer.

    Args:
        in_axis (int): Input dimension axis.
        out_axis (int): Output dimension axis.
        batch_axis (tuple[int, ...]): Batch axes.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: He normal initializer callable.
    """
    return variance_scaling(
        InitializerConfig(
            scale=2.0,
            mode="fan_in",
            distribution="truncated_normal",
            in_axis=in_axis,
            out_axis=out_axis,
            batch_axis=batch_axis,
            dtype=_resolve_dtype(dtype),
        )
    )


kaiming_normal = he_normal


def orthogonal(
    scale: float = 1.0,
    column_axis: int = -1,
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build an orthogonal matrix initializer via QR decomposition.

    Args:
        scale (float): Multiplicative scale factor applied to the orthogonal matrix.
        column_axis (int): Axis corresponding to the columns of the orthogonal matrix.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: Orthogonal initializer callable.
    """

    def init(
        key: Tensor | int | None,
        shape: tuple[int, ...],
        init_dtype: dtypes.DType | str | None = None,
    ) -> Tensor:
        """Generate an orthogonal or semi-orthogonal tensor via QR decomposition.

        Args:
            key (Tensor | int | None): PRNG key or seed.
            shape (tuple[int, ...]): Output tensor shape.
            init_dtype (dtypes.DType | str | None): Optional override data type.

        Returns:
            Tensor: Orthogonally initialized tensor.
        """
        dt = _resolve_dtype(init_dtype, dtype)
        if len(shape) < 2:
            return normal(stddev=scale, dtype=dt)(key, shape, dt)
        col_ax = column_axis % len(shape)
        cols = shape[col_ax]
        rows = 1
        for idx, dim in enumerate(shape):
            if idx != col_ax:
                rows *= dim
        flat_matrix = ops.randn(max(rows, cols), min(rows, cols), dtype=dt)
        q_mat, _ = getattr(ops, "qr")(flat_matrix)
        reshaped = getattr(ops, "reshape")(q_mat, shape)
        return ops.scalar_mul(reshaped, float(scale))

    return init


def delta_orthogonal(
    scale: float = 1.0,
    column_axis: int = -1,
    dtype: dtypes.DType | str = dtypes.DType.Float32,
) -> InitializerFn:
    """Build a delta-orthogonal initializer for convolutional kernels.

    Args:
        scale (float): Multiplicative scale factor.
        column_axis (int): Column axis of the orthogonal submatrix.
        dtype (dtypes.DType | str): Default data type.

    Returns:
        InitializerFn: Delta-orthogonal initializer callable.
    """
    return orthogonal(scale=scale, column_axis=column_axis, dtype=dtype)
