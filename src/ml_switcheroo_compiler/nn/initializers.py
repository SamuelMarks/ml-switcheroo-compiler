"""Neural network parameter initializers."""

from __future__ import annotations

import math
import typing
from collections.abc import Sequence

import ml_switcheroo_compiler.core.dtype as dtypes
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import InitializerConfig as InitializerConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.qr import qr as _qr_op
from ml_switcheroo_compiler.ops.shape.api import reshape as _reshape_op


class InitializerFn(typing.Protocol):
    """Protocol for neural network weight initializer callables."""

    def __call__(
        self,
        key: Tensor | int | None,
        shape: tuple[int, ...],
        init_dtype: dtypes.DType | str | None = None,
    ) -> Tensor:
        """Sample or construct an initialized parameter tensor.

        Args:
            key (Tensor | int | None): PRNG key or integer seed.
            shape (tuple[int, ...]): Output tensor shape tuple.
            init_dtype (dtypes.DType | str | None): Optional override data type.

        Returns:
            Tensor: Initialized tensor.
        """
        ...


def _as_init_tensor(val: OpValue, shape: tuple[int, ...], dt: dtypes.DType) -> Tensor:
    """Ensure ``val`` is wrapped as a ``Tensor`` instance.

    Args:
        val (OpValue): Evaluated tensor or raw backend value.
        shape (tuple[int, ...]): Expected tensor shape.
        dt (dtypes.DType): Expected tensor data type.

    Returns:
        Tensor: ``val`` if already a ``Tensor``, otherwise wrapped in ``Tensor``.
    """
    return val if isinstance(val, Tensor) else Tensor(val, TensorConfig(shape, dt, "cpu"))


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
        default_dt: dtypes.DType | str = (
            config.dtype if isinstance(config.dtype, (dtypes.DType, str)) else dtypes.DType.Float32
        )
        dt = _resolve_dtype(init_dtype, default_dt)
        in_ax = int(config.in_axis) if isinstance(config.in_axis, int) else -2
        out_ax = int(config.out_axis) if isinstance(config.out_axis, int) else -1
        fan_in, fan_out = _compute_fans(shape, in_ax, out_ax)
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


@typing.runtime_checkable
class _QrNamespaceProtocol(typing.Protocol):
    """Protocol for an operations namespace providing ``qr`` decomposition."""

    def qr(self, a: Tensor) -> OpValue:
        """Compute QR decomposition of a 2-D matrix tensor.

        Args:
            a (Tensor): Input 2-D matrix tensor.

        Returns:
            OpValue: Tuple of ``(Q, R)`` factors.
        """
        ...


@typing.runtime_checkable
class _ReshapeNamespaceProtocol(typing.Protocol):
    """Protocol for an operations namespace providing ``reshape``."""

    def reshape(self, a: Tensor, shape: tuple[int, ...]) -> OpValue:
        """Reshape a tensor to ``shape``.

        Args:
            a (Tensor): Input tensor.
            shape (tuple[int, ...]): Target shape tuple.

        Returns:
            OpValue: Reshaped tensor.
        """
        ...


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
        mat_shape = (max(rows, cols), min(rows, cols))
        mat_tensor = _as_init_tensor(ops.randn(*mat_shape, dtype=dt), mat_shape, dt)
        ops_ns: OpValue | _QrNamespaceProtocol | _ReshapeNamespaceProtocol = ops
        qr_out: OpValue = (
            ops_ns.qr(mat_tensor)
            if isinstance(ops_ns, _QrNamespaceProtocol) and callable(ops_ns.qr)
            else _qr_op(mat_tensor)
        )
        q_raw: OpValue = qr_out[0] if isinstance(qr_out, (tuple, list)) else qr_out
        q_tensor = _as_init_tensor(q_raw, mat_shape, dt)
        reshaped = (
            ops_ns.reshape(q_tensor, shape)
            if isinstance(ops_ns, _ReshapeNamespaceProtocol) and callable(ops_ns.reshape)
            else _reshape_op(q_tensor, shape)
        )
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
