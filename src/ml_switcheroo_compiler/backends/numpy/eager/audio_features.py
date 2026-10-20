"""Extra audio ops for eager numpy execution."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


def _dct_1d_matrix(N: int, type: int, norm: str | None = None) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager dct 1d matrix operation on the active backend array.

    Args:
        N (int): The N parameter.
        type (int): The type parameter.
        norm (str): The norm parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    n: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.arange(N, dtype=float)
    k: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.arange(N, dtype=float)[:, None]
    if type == 1:
        m: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.cos(np.pi * k * n / (N - 1))
        if norm == "ortho":
            m[0, :] *= np.sqrt(0.5)
            m[-1, :] *= np.sqrt(0.5)
            m[:, 0] *= np.sqrt(0.5)
            m[:, -1] *= np.sqrt(0.5)
            m *= np.sqrt(2 / (N - 1))
        return m
    elif type == 2:
        m = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        if norm == "ortho":
            m[0, :] *= np.sqrt(0.5)
            m *= np.sqrt(2 / N)
        else:
            m *= 2.0
        return m
    elif type == 3:
        m = np.cos(np.pi * (2 * k + 1) * n / (2 * N))
        if norm == "ortho":
            m[:, 0] *= np.sqrt(0.5)
            m *= np.sqrt(2 / N)
        else:
            m[:, 0] *= 0.5
            m *= 2.0
        return m
    elif type == 4:
        m = np.cos(np.pi * (2 * k + 1) * (2 * n + 1) / (4 * N))
        if norm == "ortho":
            m *= np.sqrt(2 / N)
        else:
            m *= 2.0
        return m
    raise ValueError("Unsupported DCT type")


@numpy_eager_registry.register("Dct")
def _np_dct(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np dct operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(args[0])
    type: int = typing.cast(int, kwargs.get("type", 2))
    axis: int = typing.cast(int, kwargs.get("axis", -1))
    norm: str | None = typing.cast("str | None", kwargs.get("norm", None))

    N = x.shape[axis]
    m: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = _dct_1d_matrix(N, type, norm)
    res: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "tensordot")(x, m, axes=(axis, 1))
    res = np.moveaxis(res, -1, axis)
    return res


def _idct_1d_matrix(N: int, type: int, norm: str | None = None) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager idct 1d matrix operation on the active backend array.

    Args:
        N (int): The N parameter.
        type (int): The type parameter.
        norm (str): The norm parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    if type == 1:
        m: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = _dct_1d_matrix(N, 1, norm)
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


@numpy_eager_registry.register("Idct")
def _np_idct(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np idct operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(args[0])
    type: int = typing.cast(int, kwargs.get("type", 2))
    axis: int = typing.cast(int, kwargs.get("axis", -1))
    norm: str | None = typing.cast("str | None", kwargs.get("norm", None))

    N = x.shape[axis]
    m: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = _idct_1d_matrix(N, type, norm)
    res: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "tensordot")(x, m, axes=(axis, 1))
    res = np.moveaxis(res, -1, axis)
    return res


@numpy_eager_registry.register("Mdct")
def _np_mdct(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement MDCT eagerly in numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(args[0])
    N = kwargs.get("N", x.shape[-1] // 2)

    n_int = int(typing.cast(int, N))
    if int(x.shape[-1]) != 2 * n_int:
        raise ValueError("MDCT input last dimension must be 2N")

    n: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.arange(2 * n_int, dtype=float)
    k: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.arange(n_int, dtype=float)[:, None]
    m: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = getattr(np, "cos")(np.pi / n_int * (n + 0.5 + n_int / 2) * (k + 0.5))
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = getattr(np, "tensordot")(x, m, axes=(-1, 1))
    return res


@numpy_eager_registry.register("InverseMdct")
def _np_inverse_mdct(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Inverse MDCT eagerly in numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(args[0])
    N = x.shape[-1]

    k: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.arange(N, dtype=float)
    n: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.arange(2 * N, dtype=float)[:, None]
    m: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.cos(np.pi / N * (n + 0.5 + N / 2) * (k + 0.5))
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = getattr(np, "tensordot")(x, m, axes=(-1, 1))
    return res * (1.0 / N)


@numpy_eager_registry.register("Frame")
def _np_frame(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Frame eagerly in numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(args[0])
    frame_length: int = typing.cast(int, kwargs.get("frame_length", 1))
    frame_step: int = typing.cast(int, kwargs.get("frame_step", 1))
    num_frames: int = (x.shape[-1] - frame_length) // frame_step + 1
    if num_frames <= 0:
        return np.empty(list(x.shape)[:-1] + [0, frame_length], dtype=x.dtype)
    shape: list[int] = list(x.shape)[:-1] + [num_frames, frame_length]
    strides: list[int] = list(x.strides)[:-1] + [x.strides[-1] * frame_step, x.strides[-1]]
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


@numpy_eager_registry.register("OverlapAndAdd")
def _np_overlap_and_add(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement OverlapAndAdd eagerly in numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(args[0])
    frame_step: int = typing.cast(int, kwargs.get("frame_step", 1))
    num_frames: int = x.shape[-2]
    frame_length: int = x.shape[-1]
    res_length: int = (num_frames - 1) * frame_step + frame_length
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.zeros(list(x.shape)[:-2] + [res_length], dtype=x.dtype)

    orig_shape: tuple[int, ...] = x.shape
    batch_shape: tuple[int, ...] = orig_shape[:-2]
    flat_batch_size: int = int(np.prod(batch_shape)) if batch_shape else 1

    flat_signal: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = x.reshape((flat_batch_size, num_frames, frame_length))
    flat_res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = res.reshape((flat_batch_size, res_length))

    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_length
        flat_res[:, start:end] = flat_res[:, start:end] + flat_signal[:, i, :].astype(np.float64)

    return res
