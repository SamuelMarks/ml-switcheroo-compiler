# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

import builtins
import typing
from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Fft")
def _fft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fft operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Rfft")
def _rfft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _rfft operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.rfft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fft2")
def _fft2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fft2 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fft2(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftconvolve")
def _fftconvolve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftconvolve operation using backend or native FFT/IFFT fallback.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args (in1, in2).
        **kwargs: Keyword args (mode, axes).

    Returns:
        Any: Discrete convolution result of in1 and in2.
    """
    if len(args) < 2 or args[0] is None or args[1] is None:
        return None

    in1, in2 = args[0], args[1]
    mode = kwargs.get("mode", "full")
    axes = kwargs.get("axes", None)

    fft_mod = getattr(backend_module, "fft", None)
    if hasattr(fft_mod, "fftconvolve"):
        return fft_mod.fftconvolve(*args, **kwargs)
    if hasattr(backend_module, "signal") and hasattr(backend_module.signal, "fftconvolve"):
        return backend_module.signal.fftconvolve(*args, **kwargs)

    import numpy as np

    np_mod = np if backend_module is None else getattr(backend_module, "numpy", np)
    a1 = np_mod.asarray(in1)
    a2 = np_mod.asarray(in2)

    ndim = max(a1.ndim, a2.ndim)
    s1 = list(a1.shape) + [1] * (ndim - a1.ndim)
    s2 = list(a2.shape) + [1] * (ndim - a2.ndim)
    conv_shape = [s1[i] + s2[i] - 1 for i in range(ndim)]

    is_real = not (np_mod.iscomplexobj(a1) or np_mod.iscomplexobj(a2))
    fft_lib = getattr(np_mod, "fft", np.fft)

    if is_real:
        f1 = fft_lib.rfftn(a1, s=conv_shape, axes=axes)
        f2 = fft_lib.rfftn(a2, s=conv_shape, axes=axes)
        out = fft_lib.irfftn(f1 * f2, s=conv_shape, axes=axes)
    else:
        f1 = fft_lib.fftn(a1, s=conv_shape, axes=axes)
        f2 = fft_lib.fftn(a2, s=conv_shape, axes=axes)
        out = fft_lib.ifftn(f1 * f2, s=conv_shape, axes=axes)

    if mode == "full":
        return out
    elif mode == "same":
        slices = []
        for i in range(ndim):
            start = (s2[i] - 1) // 2
            slices.append(slice(start, start + s1[i]))
        return out[tuple(slices)]
    elif mode == "valid":
        slices = []
        for i in range(ndim):
            start = s2[i] - 1
            length = s1[i] - s2[i] + 1
            if length < 1:
                return np_mod.empty((0,) * ndim, dtype=out.dtype)
            slices.append(slice(start, start + length))
        return out[tuple(slices)]

    return out


@global_eager_registry.register("Fftfreq")
def _fftfreq(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftfreq operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fftfreq(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftn")
def _fftn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftn operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftnd")
def _fftnd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftnd operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    if hasattr(fft_mod, "fftnd"):
        return fft_mod.fftnd(*args, **kwargs)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftshift")
def _fftshift(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftshift operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fftshift(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifft")
def _ifft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ifft operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.ifft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifft2")
def _ifft2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ifft2 operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.ifft2(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifftn")
def _ifftn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ifftn operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.ifftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifftshift")
def _ifftshift(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ifftshift operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.ifftshift(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("NpHfft")
def _np_hfft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_hfft operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    if fft_mod and hasattr(fft_mod, "hfft"):
        return fft_mod.hfft(*args, **kwargs)
    import numpy as np

    return np.fft.hfft(*args, **kwargs)
