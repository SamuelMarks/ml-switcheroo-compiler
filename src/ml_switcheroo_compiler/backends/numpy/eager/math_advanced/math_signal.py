# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_signal module."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_general import _get_np_arg


@numpy_eager_registry.register("Correlate")
def _np_correlate(backend_module, *args, **kwargs):
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.correlate(*args, **kwargs)


@numpy_eager_registry.register("Blackman")
def _np_blackman_(backend_module, *args, **kwargs):
    """Implement Blackman via blackman.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.blackman(*args, **kwargs)


@numpy_eager_registry.register("Hamming")
def _np_hamming_(backend_module, *args, **kwargs):
    """Implement Hamming via hamming.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.hamming(*args, **kwargs)


@numpy_eager_registry.register("Hanning")
def _np_hanning_(backend_module, *args, **kwargs):
    """Implement Hanning via hanning.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.hanning(*args, **kwargs)


@numpy_eager_registry.register("Kaiser")
def _np_kaiser_(backend_module, *args, **kwargs):
    """Implement Kaiser via kaiser.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.kaiser(*args, **kwargs)


@numpy_eager_registry.register("Rfft")
def _np_rfft(backend_module, *args, **kwargs):
    """Evaluate _np_rfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.rfft(a, **kwargs)


@numpy_eager_registry.register("Ifft")
def _np_ifft(backend_module, *args, **kwargs):
    """Evaluate _np_ifft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifft(a, **kwargs)


@numpy_eager_registry.register("Fftn")
def _np_fftn(backend_module, *args, **kwargs):
    """Evaluate _np_fftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.fftn(a, **kwargs)


@numpy_eager_registry.register("Ifftn")
def _np_ifftn(backend_module, *args, **kwargs):
    """Evaluate _np_ifftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifftn(a, **kwargs)


@numpy_eager_registry.register("Rfftn")
def _np_rfftn(backend_module, *args, **kwargs):
    """Evaluate _np_rfftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.rfftn(a, **kwargs)


@numpy_eager_registry.register("Irfftn")
def _np_irfftn(backend_module, *args, **kwargs):
    """Evaluate _np_irfftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.irfftn(a, **kwargs)


@numpy_eager_registry.register("Ifft2")
def _np_ifft2(backend_module, *args, **kwargs):
    """Evaluate _np_ifft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifft2(a, **kwargs)


@numpy_eager_registry.register("Rfft2")
def _np_rfft2(backend_module, *args, **kwargs):
    """Evaluate _np_rfft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.rfft2(a, **kwargs)


@numpy_eager_registry.register("Irfft2")
def _np_irfft2(backend_module, *args, **kwargs):
    """Evaluate _np_irfft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.irfft2(a, **kwargs)


@numpy_eager_registry.register("Fftnd")
def _np_fftnd(backend_module, *args, **kwargs):
    """Evaluate _np_fftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.fftn(a, **kwargs)


@numpy_eager_registry.register("Ifftnd")
def _np_ifftnd(backend_module, *args, **kwargs):
    """Evaluate _np_ifftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifftn(a, **kwargs)


@numpy_eager_registry.register("Rfftnd")
def _np_rfftnd(backend_module, *args, **kwargs):
    """Evaluate _np_rfftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.rfftn(a, **kwargs)


@numpy_eager_registry.register("Irfftnd")
def _np_irfftnd(backend_module, *args, **kwargs):
    """Evaluate _np_irfftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.irfftn(a, **kwargs)


@numpy_eager_registry.register("Fftshift")
def _np_fftshift(backend_module, *args, **kwargs):
    """Evaluate _np_fftshift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.fftshift(a, **kwargs)


@numpy_eager_registry.register("Ifftshift")
def _np_ifftshift(backend_module, *args, **kwargs):
    """Evaluate _np_ifftshift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifftshift(a, **kwargs)


@numpy_eager_registry.register("Hfft")
def _np_hfft(backend_module, *args, **kwargs):
    """Evaluate _np_hfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.hfft(a, **kwargs)


@numpy_eager_registry.register("Rfftfreq")
def _np_rfftfreq(backend_module, *args, **kwargs):
    """Evaluate _np_rfftfreq operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy.fft as fft

    if not args:
        return None
    try:
        n = int(args[0])
    except (TypeError, ValueError):
        return args[0]
    return fft.rfftfreq(n, **kwargs)


@numpy_eager_registry.register("Rrelu")
def _np_rrelu(backend_module, *args, **kwargs):
    """Evaluate _np_rrelu operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    lower = kwargs.get("lower", 0.125)
    upper = kwargs.get("upper", 0.333)
    alpha = np.random.uniform(lower, upper, size=a.shape)
    return np.where(a >= 0, a, a * alpha)


@numpy_eager_registry.register("Softmax")
def _np_softmax(backend_module, *args, **kwargs):
    """Evaluate _np_softmax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    axis = kwargs.get("axis", -1)
    e_x = np.exp(a - np.max(a, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


@numpy_eager_registry.register("Sigmoid")
def _np_sigmoid(backend_module, *args, **kwargs):
    """Evaluate _np_sigmoid operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return 1.0 / (1.0 + np.exp(-a))
