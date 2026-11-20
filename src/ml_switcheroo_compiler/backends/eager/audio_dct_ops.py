"""Generic backend-parametric eager reference operations for `audio_dct_ops`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_int,
    coerce_opt_str,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


def _dct_1d_matrix(N: int, type: int, norm: str | None = None) -> EagerTensorProtocol:
    """Execute the eager dct 1d matrix operation on the active backend array.

    Args:
        N (int): Input `N` argument (N).
        type (int): Input `type` argument (type).
        norm (str | None): Input `norm` argument (norm).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    arr_mod = _resolve_eager_mod(None)
    n: EagerTensorProtocol = arr_mod.arange(N, dtype=float)
    k: EagerTensorProtocol = arr_mod.arange(N, dtype=float)[:, None]
    if type == 1:
        m: EagerTensorProtocol = arr_mod.cos(arr_mod.pi * k * n / (N - 1))
        if norm == "ortho":
            m[0, :] *= arr_mod.sqrt(0.5)
            m[-1, :] *= arr_mod.sqrt(0.5)
            m[:, 0] *= arr_mod.sqrt(0.5)
            m[:, -1] *= arr_mod.sqrt(0.5)
            m *= arr_mod.sqrt(2 / (N - 1))
        return m
    elif type == 2:
        m = arr_mod.cos(arr_mod.pi * k * (2 * n + 1) / (2 * N))
        if norm == "ortho":
            m[0, :] *= arr_mod.sqrt(0.5)
            m *= arr_mod.sqrt(2 / N)
        else:
            m *= 2.0
        return m
    elif type == 3:
        m = arr_mod.cos(arr_mod.pi * (2 * k + 1) * n / (2 * N))
        if norm == "ortho":
            m[:, 0] *= arr_mod.sqrt(0.5)
            m *= arr_mod.sqrt(2 / N)
        else:
            m[:, 0] *= 0.5
            m *= 2.0
        return m
    elif type == 4:
        m = arr_mod.cos(arr_mod.pi * (2 * k + 1) * (2 * n + 1) / (4 * N))
        if norm == "ortho":
            m *= arr_mod.sqrt(2 / N)
        else:
            m *= 2.0
        return m
    raise ValueError("Unsupported DCT type")


@global_eager_registry.register("Dct")
def _np_dct(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np dct operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x: EagerTensorProtocol = arr_mod.asarray(args[0])
    type: int = coerce_int(kwargs.get("type", 2))
    axis: int = coerce_int(kwargs.get("axis", -1))
    norm: str | None = coerce_opt_str(kwargs.get("norm", None))

    N = x.shape[axis]
    m: EagerTensorProtocol = _dct_1d_matrix(N, type, norm)
    res: EagerTensorProtocol = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("tensordot")(x, m, axes=(axis, 1))
    )
    res = arr_mod.moveaxis(res, -1, axis)
    return res


def _idct_1d_matrix(N: int, type: int, norm: str | None = None) -> EagerTensorProtocol:
    """Execute the eager idct 1d matrix operation on the active backend array.

    Args:
        N (int): Input `N` argument (N).
        type (int): Input `type` argument (type).
        norm (str | None): Input `norm` argument (norm).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    if type == 1:
        m: EagerTensorProtocol = _dct_1d_matrix(N, 1, norm)
        if norm != "ortho":
            m /= 2 * (N - 1)
        return m
    elif type == 2:
        m = _dct_1d_matrix(N, 3, norm)
        if norm != "ortho":
            m /= 2 * N
        return m
    elif type == 3:
        m = _dct_1d_matrix(N, 2, norm)
        if norm != "ortho":
            m /= 2 * N
        return m
    elif type == 4:
        m = _dct_1d_matrix(N, 4, norm)
        if norm != "ortho":
            m /= 2 * N
        return m
    raise ValueError("Unsupported IDCT type")


@global_eager_registry.register("Idct")
def _np_idct(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np idct operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x: EagerTensorProtocol = arr_mod.asarray(args[0])
    type: int = coerce_int(kwargs.get("type", 2))
    axis: int = coerce_int(kwargs.get("axis", -1))
    norm: str | None = coerce_opt_str(kwargs.get("norm", None))

    N = x.shape[axis]
    m: EagerTensorProtocol = _idct_1d_matrix(N, type, norm)
    res: EagerTensorProtocol = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("tensordot")(x, m, axes=(axis, 1))
    )
    res = arr_mod.moveaxis(res, -1, axis)
    return res


@global_eager_registry.register("Mdct")
def _np_mdct(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement MDCT eagerly in numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x: EagerTensorProtocol = arr_mod.asarray(args[0])
    N = kwargs.get("N", x.shape[-1] // 2)

    n_int = int(coerce_int(N))
    if int(x.shape[-1]) != 2 * n_int:
        raise ValueError("MDCT input last dimension must be 2N")

    n: EagerTensorProtocol = arr_mod.arange(2 * n_int, dtype=float)
    k: EagerTensorProtocol = arr_mod.arange(n_int, dtype=float)[:, None]
    m: EagerTensorProtocol = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("cos")(arr_mod.pi / n_int * (n + 0.5 + n_int / 2) * (k + 0.5))
    )
    res: EagerTensorProtocol = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("tensordot")(x, m, axes=(-1, 1))
    )
    return res


@global_eager_registry.register("InverseMdct")
def _np_inverse_mdct(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Inverse MDCT eagerly in numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x: EagerTensorProtocol = arr_mod.asarray(args[0])
    N = x.shape[-1]

    k: EagerTensorProtocol = arr_mod.arange(N, dtype=float)
    n: EagerTensorProtocol = arr_mod.arange(2 * N, dtype=float)[:, None]
    m: EagerTensorProtocol = arr_mod.cos(arr_mod.pi / N * (n + 0.5 + N / 2) * (k + 0.5))
    res: EagerTensorProtocol = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("tensordot")(x, m, axes=(-1, 1))
    )
    return res * (1.0 / N)


@global_eager_registry.register("Frame")
def _np_frame(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Frame eagerly in numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x: EagerTensorProtocol = arr_mod.asarray(args[0])
    frame_length: int = coerce_int(kwargs.get("frame_length", 1))
    frame_step: int = coerce_int(kwargs.get("frame_step", 1))
    num_frames: int = (x.shape[-1] - frame_length) // frame_step + 1
    if num_frames <= 0:
        return arr_mod.empty(list(x.shape)[:-1] + [0, frame_length], dtype=x.dtype)
    shape: list[int] = list(x.shape)[:-1] + [num_frames, frame_length]
    x_strides = [int(coerce_int(st)) for st in adapt_backend_module(x).require_symbol("strides")]
    strides: list[int] = x_strides[:-1] + [x_strides[-1] * frame_step, x_strides[-1]]
    return arr_mod.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


@global_eager_registry.register("OverlapAndAdd")
def _np_overlap_and_add(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement OverlapAndAdd eagerly in numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x: EagerTensorProtocol = arr_mod.asarray(args[0])
    frame_step: int = coerce_int(kwargs.get("frame_step", 1))
    num_frames: int = x.shape[-2]
    frame_length: int = x.shape[-1]
    res_length: int = (num_frames - 1) * frame_step + frame_length
    res: EagerTensorProtocol = arr_mod.zeros(list(x.shape)[:-2] + [res_length], dtype=x.dtype)

    orig_shape: tuple[int, ...] = x.shape
    batch_shape: tuple[int, ...] = orig_shape[:-2]
    flat_batch_size: int = int(arr_mod.prod(batch_shape)) if batch_shape else 1

    flat_signal: EagerTensorProtocol = x.reshape((flat_batch_size, num_frames, frame_length))
    flat_res: EagerTensorProtocol = res.reshape((flat_batch_size, res_length))

    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_length
        flat_res[:, start:end] = flat_res[:, start:end] + flat_signal[:, i, :].astype(arr_mod.float64)

    return res
