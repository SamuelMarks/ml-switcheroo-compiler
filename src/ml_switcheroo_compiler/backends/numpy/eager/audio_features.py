# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Extra audio ops for eager numpy execution."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _dct_1d_matrix(N: int, type: int, norm: object = None) -> np.ndarray:
    """Evaluate _dct_1d_matrix operation.

    Args:
        N (int): The N parameter.
        type (int): The type parameter.
        norm (str): The norm parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    n: object = np.arange(N)
    k: object = np.arange(N)[:, None]
    if type == 1:
        m: object = np.cos(np.pi * k * n / (N - 1))
        if norm == "ortho":
            m[0, :] *= np.sqrt(0.5)
            m[-1, :] *= np.sqrt(0.5)
            m[:, 0] *= np.sqrt(0.5)
            m[:, -1] *= np.sqrt(0.5)
            m *= np.sqrt(2 / (N - 1))
        return m
    elif type == 2:
        m: object = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        if norm == "ortho":
            m[0, :] *= np.sqrt(0.5)
            m *= np.sqrt(2 / N)
        else:
            m *= 2.0
        return m
    elif type == 3:
        m: object = np.cos(np.pi * (2 * k + 1) * n / (2 * N))
        if norm == "ortho":
            m[:, 0] *= np.sqrt(0.5)
            m *= np.sqrt(2 / N)
        else:
            m[:, 0] *= 0.5
            m *= 2.0
        return m
    elif type == 4:
        m: object = np.cos(np.pi * (2 * k + 1) * (2 * n + 1) / (4 * N))
        if norm == "ortho":
            m *= np.sqrt(2 / N)
        else:
            m *= 2.0
        return m
    raise ValueError("Unsupported DCT type")


@numpy_eager_registry.register("Dct")
def _np_dct(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_dct operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: object = np.asarray(args[0])
    type: object = kwargs.get("type", 2)
    axis: object = kwargs.get("axis", -1)
    norm: object = kwargs.get("norm", None)

    N = x.shape[axis]
    m: object = _dct_1d_matrix(N, type, norm)
    res: object = np.tensordot(x, m, axes=(axis, 1))
    res: object = np.moveaxis(res, -1, axis)
    return res


def _idct_1d_matrix(N: int, type: int, norm: object = None) -> np.ndarray:
    """Evaluate _idct_1d_matrix operation.

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
        m: object = _dct_1d_matrix(N, 1, norm)
        if norm != "ortho":
            m /= 2 * (N - 1)
        return m
    elif type == 2:
        m: object = _dct_1d_matrix(N, 3, norm)
        if norm != "ortho":
            m /= 2 * N
        return m
    elif type == 3:
        m: object = _dct_1d_matrix(N, 2, norm)
        if norm != "ortho":
            m /= 2 * N
        return m
    elif type == 4:
        m: object = _dct_1d_matrix(N, 4, norm)
        if norm != "ortho":
            m /= 2 * N
        return m
    raise ValueError("Unsupported IDCT type")


@numpy_eager_registry.register("Idct")
def _np_idct(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_idct operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: object = np.asarray(args[0])
    type: object = kwargs.get("type", 2)
    axis: object = kwargs.get("axis", -1)
    norm: object = kwargs.get("norm", None)

    N = x.shape[axis]
    m: object = _idct_1d_matrix(N, type, norm)
    res: object = np.tensordot(x, m, axes=(axis, 1))
    res: object = np.moveaxis(res, -1, axis)
    return res


@numpy_eager_registry.register("Mdct")
def _np_mdct(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement MDCT eagerly in numpy.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    x: object = np.asarray(args[0])
    N = kwargs.get("N", x.shape[-1] // 2)

    if x.shape[-1] != 2 * N:
        raise ValueError("MDCT input last dimension must be 2N")

    n: object = np.arange(2 * N)
    k: object = np.arange(N)[:, None]
    m: object = np.cos(np.pi / N * (n + 0.5 + N / 2) * (k + 0.5))
    res: object = np.tensordot(x, m, axes=(-1, 1))
    return res


@numpy_eager_registry.register("InverseMdct")
def _np_inverse_mdct(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Inverse MDCT eagerly in numpy.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: object = np.asarray(args[0])
    N = x.shape[-1]

    k: object = np.arange(N)
    n: object = np.arange(2 * N)[:, None]
    m: object = np.cos(np.pi / N * (n + 0.5 + N / 2) * (k + 0.5))
    res: object = np.tensordot(x, m, axes=(-1, 1))
    return res * (1.0 / N)


@numpy_eager_registry.register("Frame")
def _np_frame(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Frame eagerly in numpy.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: object = np.asarray(args[0])
    frame_length: object = kwargs.get("frame_length", 1)
    frame_step: object = kwargs.get("frame_step", 1)
    num_frames: object = (x.shape[-1] - frame_length) // frame_step + 1
    if num_frames <= 0:
        return np.empty(list(x.shape)[:-1] + [0, frame_length], dtype=x.dtype)
    shape: object = list(x.shape)[:-1] + [num_frames, frame_length]
    strides: object = list(x.strides)[:-1] + [x.strides[-1] * frame_step, x.strides[-1]]
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


@numpy_eager_registry.register("OverlapAndAdd")
def _np_overlap_and_add(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement OverlapAndAdd eagerly in numpy.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    x: object = np.asarray(args[0])
    frame_step: object = kwargs.get("frame_step", 1)
    num_frames: object = x.shape[-2]
    frame_length: object = x.shape[-1]
    res_length: object = (num_frames - 1) * frame_step + frame_length
    res: object = np.zeros(list(x.shape)[:-2] + [res_length], dtype=x.dtype)

    orig_shape: object = x.shape
    batch_shape: object = orig_shape[:-2]
    flat_batch_size: object = int(np.prod(batch_shape)) if batch_shape else 1

    flat_signal: object = x.reshape((flat_batch_size, num_frames, frame_length))
    flat_res: object = res.reshape((flat_batch_size, res_length))

    for i in range(num_frames):
        start: object = i * frame_step
        end: object = start + frame_length
        flat_res[:, start:end] += flat_signal[:, i, :]

    return res
