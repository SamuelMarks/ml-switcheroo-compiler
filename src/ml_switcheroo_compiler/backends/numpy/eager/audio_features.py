# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Extra audio ops for eager numpy execution."""

from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _dct_1d_matrix(N: int, type: int, norm: Any = None) -> np.ndarray:
    """Evaluate _dct_1d_matrix operation.

    Args:
        N (int): The N parameter.
        type (int): The type parameter.
        norm (str): The norm parameter.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    n = np.arange(N)
    k = np.arange(N)[:, None]
    if type == 1:
        m = np.cos(np.pi * k * n / (N - 1))
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
def _np_dct(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_dct operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = np.asarray(args[0])
    type = kwargs.get("type", 2)
    axis = kwargs.get("axis", -1)
    norm = kwargs.get("norm", None)

    N = x.shape[axis]
    m = _dct_1d_matrix(N, type, norm)
    res = np.tensordot(x, m, axes=(axis, 1))
    res = np.moveaxis(res, -1, axis)
    return res


def _idct_1d_matrix(N: int, type: int, norm: Any = None) -> np.ndarray:
    """Evaluate _idct_1d_matrix operation.

    Args:
        N (int): The N parameter.
        type (int): The type parameter.
        norm (str): The norm parameter.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    if type == 1:
        m = _dct_1d_matrix(N, 1, norm)
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
def _np_idct(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_idct operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = np.asarray(args[0])
    type = kwargs.get("type", 2)
    axis = kwargs.get("axis", -1)
    norm = kwargs.get("norm", None)

    N = x.shape[axis]
    m = _idct_1d_matrix(N, type, norm)
    res = np.tensordot(x, m, axes=(axis, 1))
    res = np.moveaxis(res, -1, axis)
    return res


@numpy_eager_registry.register("Mdct")
def _np_mdct(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement MDCT eagerly in numpy.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x = np.asarray(args[0])
    N = kwargs.get("N", x.shape[-1] // 2)

    if x.shape[-1] != 2 * N:
        raise ValueError("MDCT input last dimension must be 2N")

    n = np.arange(2 * N)
    k = np.arange(N)[:, None]
    m = np.cos(np.pi / N * (n + 0.5 + N / 2) * (k + 0.5))
    res = np.tensordot(x, m, axes=(-1, 1))
    return res


@numpy_eager_registry.register("InverseMdct")
def _np_inverse_mdct(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Inverse MDCT eagerly in numpy.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = np.asarray(args[0])
    N = x.shape[-1]

    k = np.arange(N)
    n = np.arange(2 * N)[:, None]
    m = np.cos(np.pi / N * (n + 0.5 + N / 2) * (k + 0.5))
    res = np.tensordot(x, m, axes=(-1, 1))
    return res * (1.0 / N)


@numpy_eager_registry.register("Frame")
def _np_frame(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Frame eagerly in numpy.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = np.asarray(args[0])
    frame_length = kwargs.get("frame_length", 1)
    frame_step = kwargs.get("frame_step", 1)
    num_frames = (x.shape[-1] - frame_length) // frame_step + 1
    if num_frames <= 0:
        return np.empty(list(x.shape)[:-1] + [0, frame_length], dtype=x.dtype)
    shape = list(x.shape)[:-1] + [num_frames, frame_length]
    strides = list(x.strides)[:-1] + [x.strides[-1] * frame_step, x.strides[-1]]
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


@numpy_eager_registry.register("OverlapAndAdd")
def _np_overlap_and_add(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement OverlapAndAdd eagerly in numpy.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = np.asarray(args[0])
    frame_step = kwargs.get("frame_step", 1)
    num_frames = x.shape[-2]
    frame_length = x.shape[-1]
    res_length = (num_frames - 1) * frame_step + frame_length
    res = np.zeros(list(x.shape)[:-2] + [res_length], dtype=x.dtype)

    orig_shape = x.shape
    batch_shape = orig_shape[:-2]
    flat_batch_size = int(np.prod(batch_shape)) if batch_shape else 1

    flat_signal = x.reshape((flat_batch_size, num_frames, frame_length))
    flat_res = res.reshape((flat_batch_size, res_length))

    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_length
        flat_res[:, start:end] += flat_signal[:, i, :]

    return res
