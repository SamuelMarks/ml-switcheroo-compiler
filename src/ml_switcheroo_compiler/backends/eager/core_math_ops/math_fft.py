"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft`."""

from __future__ import annotations

import builtins  # noqa: F401
import types
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Fft")
def _fft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fft operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.fft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Rfft")
def _rfft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager rfft operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.rfft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fft2")
def _fft2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fft2 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.fft2(*args, **kwargs) if fft_mod else None


def _slice_fftconvolve_output(
    out: BackendArg,
    mode: BackendArg,
    s1: list[int],
    s2: list[int],
    ndim: int,
    np_mod: BackendModuleProtocol,
) -> BackendArg:
    """Slice full FFT convolution result according to the requested mode ('full', 'same', 'valid').

    Args:
        out (BackendArg): Full discrete convolution array.
        mode (BackendArg): Convolution mode string.
        s1 (list[int]): Padded shape of first input.
        s2 (list[int]): Padded shape of second input.
        ndim (int): Number of dimensions.
        np_mod (BackendModuleProtocol): NumPy-compatible array module.

    Returns:
        BackendArg: Sliced convolution output array.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    out_arr = coerce_array(np_mod, out)
    if mode == "same":
        slices = [slice((s2[i] - 1) // 2, (s2[i] - 1) // 2 + s1[i]) for i in range(ndim)]
        return out_arr[tuple(slices)]
    if mode == "valid":
        slices = []
        for i in range(ndim):
            start = s2[i] - 1
            length = s1[i] - s2[i] + 1
            if length < 1:
                return adapt_backend_module(np_mod).require_symbol("empty")(
                    (0,) * ndim, dtype=adapt_backend_module(out).require_symbol("dtype")
                )
            slices.append(slice(start, start + length))
        return out_arr[tuple(slices)]
    return out


@global_eager_registry.register("Fftconvolve")
def _fftconvolve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate _fftconvolve operation using backend or native FFT/IFFT fallback.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args (in1, in2).
        **kwargs: Keyword args (mode, axes).

    Returns:
        BackendArg: Discrete convolution result of in1 and in2.
    """
    if len(args) < 2 or args[0] is None or args[1] is None:
        return None

    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    if adapt_backend_module(fft_mod).has_symbol("fftconvolve"):
        return adapt_backend_module(fft_mod).require_symbol("fftconvolve")(*args, **kwargs)
    signal_mod = adapt_backend_module(backend_module).get_symbol("signal", None)
    if adapt_backend_module(signal_mod).has_symbol("fftconvolve"):
        return adapt_backend_module(signal_mod).require_symbol("fftconvolve")(*args, **kwargs)

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

    a1_nd = a1.reshape(s1)
    a2_nd = a2.reshape(s2)
    resolved_axes = tuple(range(ndim)) if axes is None else axes
    is_real = not (np_mod.iscomplexobj(a1_nd) or np_mod.iscomplexobj(a2_nd))
    fft_lib = adapt_backend_module(np_mod).require_symbol("fft")

    if is_real:
        f1 = fft_lib.rfftn(a1_nd, s=conv_shape, axes=resolved_axes)
        f2 = fft_lib.rfftn(a2_nd, s=conv_shape, axes=resolved_axes)
        out = fft_lib.irfftn(f1 * f2, s=conv_shape, axes=resolved_axes)
    else:
        f1 = fft_lib.fftn(a1_nd, s=conv_shape, axes=resolved_axes)
        f2 = fft_lib.fftn(a2_nd, s=conv_shape, axes=resolved_axes)
        out = fft_lib.ifftn(f1 * f2, s=conv_shape, axes=resolved_axes)

    return _slice_fftconvolve_output(out, mode, s1, s2, ndim, np_mod)


@global_eager_registry.register("Fftfreq")
def _fftfreq(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fftfreq operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.fftfreq(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftn")
def _fftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fftn operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftnd")
def _fftnd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fftnd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    if adapt_backend_module(fft_mod).has_symbol("fftnd"):
        return adapt_backend_module(fft_mod).require_symbol("fftnd")(*args, **kwargs)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftshift")
def _fftshift(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fftshift operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.fftshift(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifft")
def _ifft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager ifft operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.ifft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifft2")
def _ifft2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager ifft2 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.ifft2(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifftn")
def _ifftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager ifftn operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.ifftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifftshift")
def _ifftshift(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager ifftshift operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    return fft_mod.ifftshift(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("NpHfft")
def _np_hfft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np hfft operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    fft_mod = adapt_backend_module(backend_module).get_symbol("fft", None)
    if fft_mod and adapt_backend_module(fft_mod).has_symbol("hfft"):
        return fft_mod.hfft(*args, **kwargs)
    return _resolve_eager_mod(backend_module).fft.hfft(*args, **kwargs)
