"""Core utilities."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("Fft")
def _fft(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _rfft(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _fft2(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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


def _slice_fftconvolve_output(
    out: EagerValue,
    mode: EagerValue,
    s1: list[int],
    s2: list[int],
    ndim: int,
    np_mod: EagerValue,
) -> EagerValue:
    """Slice full FFT convolution result according to the requested mode ('full', 'same', 'valid').

    Args:
        out (EagerValue): Full discrete convolution array.
        mode (EagerValue): Convolution mode string.
        s1 (list[int]): Padded shape of first input.
        s2 (list[int]): Padded shape of second input.
        ndim (int): Number of dimensions.
        np_mod (EagerValue): NumPy-compatible array module.

    Returns:
        EagerValue: Sliced convolution output array.
    """
    if mode == "same":
        slices = [slice((s2[i] - 1) // 2, (s2[i] - 1) // 2 + s1[i]) for i in range(ndim)]
        return typing.cast(EagerTensorProtocol, out)[tuple(slices)]
    if mode == "valid":
        slices = []
        for i in range(ndim):
            start = s2[i] - 1
            length = s1[i] - s2[i] + 1
            if length < 1:
                return cast(EagerValue, getattr(np_mod, "empty")((0,) * ndim, dtype=getattr(out, "dtype")))
            slices.append(slice(start, start + length))
        return typing.cast(EagerTensorProtocol, out)[tuple(slices)]
    return out


@global_eager_registry.register("Fftconvolve")
def _fftconvolve(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _fftconvolve operation using backend or native FFT/IFFT fallback.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args (in1, in2).
        **kwargs: Keyword args (mode, axes).

    Returns:
        EagerValue: Discrete convolution result of in1 and in2.
    """
    if len(args) < 2 or args[0] is None or args[1] is None:
        return None

    fft_mod = getattr(backend_module, "fft", None)
    if hasattr(fft_mod, "fftconvolve"):
        return cast(EagerValue, getattr(fft_mod, "fftconvolve")(*args, **kwargs))
    signal_mod = getattr(backend_module, "signal", None)
    if hasattr(signal_mod, "fftconvolve"):
        return cast(EagerValue, getattr(signal_mod, "fftconvolve")(*args, **kwargs))

    in1, in2 = args[0], args[1]
    mode = kwargs.get("mode", "full")
    axes = kwargs.get("axes", None)

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

    return _slice_fftconvolve_output(out, mode, s1, s2, ndim, np_mod)


@global_eager_registry.register("Fftfreq")
def _fftfreq(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _fftn(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _fftnd(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return cast(EagerValue, getattr(fft_mod, "fftnd")(*args, **kwargs))
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftshift")
def _fftshift(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _ifft(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _ifft2(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _ifftn(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _ifftshift(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _np_hfft(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return cast(EagerValue, fft_mod.hfft(*args, **kwargs))
    return cast(EagerValue, np.fft.hfft(*args, **kwargs))
