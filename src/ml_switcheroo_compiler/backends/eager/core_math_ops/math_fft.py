"""Implementation of math fft operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("Fft")
def _fft(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fft operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Rfft")
def _rfft(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager rfft operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.rfft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fft2")
def _fft2(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fft2 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fft2(*args, **kwargs) if fft_mod else None


def _slice_fftconvolve_output(
    out: EagerValue,
    mode: EagerValue,
    s1: list[int],
    s2: list[int],
    ndim: int,
    np_mod: BackendModuleProtocol,
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
                return typing.cast("EagerValue", getattr(np_mod, "empty")((0,) * ndim, dtype=getattr(out, "dtype")))
            slices.append(slice(start, start + length))
        return typing.cast(EagerTensorProtocol, out)[tuple(slices)]
    return out


@global_eager_registry.register("Fftconvolve")
def _fftconvolve(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return typing.cast("EagerValue", getattr(fft_mod, "fftconvolve")(*args, **kwargs))
    signal_mod = getattr(backend_module, "signal", None)
    if hasattr(signal_mod, "fftconvolve"):
        return typing.cast("EagerValue", getattr(signal_mod, "fftconvolve")(*args, **kwargs))

    in1, in2 = args[0], args[1]
    mode = kwargs.get("mode", "full")
    axes = kwargs.get("axes", None)

    np_mod = _resolve_eager_mod(backend_module)
    a1 = np_mod.asarray(in1)
    a2 = np_mod.asarray(in2)

    ndim = max(a1.ndim, a2.ndim)
    s1 = list(a1.shape) + [1] * (ndim - a1.ndim)
    s2 = list(a2.shape) + [1] * (ndim - a2.ndim)
    conv_shape = [s1[i] + s2[i] - 1 for i in range(ndim)]

    is_real = not (np_mod.iscomplexobj(a1) or np_mod.iscomplexobj(a2))
    fft_lib = getattr(np_mod, "fft")

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
def _fftfreq(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fftfreq operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fftfreq(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftn")
def _fftn(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fftn operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftnd")
def _fftnd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fftnd operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    if hasattr(fft_mod, "fftnd"):
        return typing.cast("EagerValue", getattr(fft_mod, "fftnd")(*args, **kwargs))
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftshift")
def _fftshift(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fftshift operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fftshift(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifft")
def _ifft(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ifft operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.ifft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifft2")
def _ifft2(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ifft2 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.ifft2(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifftn")
def _ifftn(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ifftn operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.ifftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifftshift")
def _ifftshift(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ifftshift operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.ifftshift(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("NpHfft")
def _np_hfft(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np hfft operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    if fft_mod and hasattr(fft_mod, "hfft"):
        return typing.cast("EagerValue", fft_mod.hfft(*args, **kwargs))
    return _resolve_eager_mod(backend_module).fft.hfft(*args, **kwargs)
