"""Generic backend-parametric eager reference operations for `reference_math.math_fft`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
    adapt_backend_module,
)


@global_eager_registry.register("Fft")
def _np_fft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fft operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").fft(*args, **kwargs)


@global_eager_registry.register("Rfft")
def _np_rfft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfft operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").rfft(*args, **kwargs)


@global_eager_registry.register("Ifft")
def _np_ifft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifft operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").ifft(*args, **kwargs)


@global_eager_registry.register("Irfft")
def _np_irfft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np irfft operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").irfft(*args, **kwargs)


def _with_default_fft_axes(
    args: tuple[BackendArg, ...],
    kwargs: dict[str, BackendArg],
) -> tuple[tuple[BackendArg, ...], dict[str, BackendArg]]:
    """Populate explicit trailing `axes` when `s` is specified without `axes` (NumPy 2.0+ requirement).

    Args:
        args (tuple[BackendArg, ...]): Positional FFT arguments `(a, [s, [axes, ...]])`.
        kwargs (dict[str, BackendArg]): Keyword FFT arguments.

    Returns:
        tuple[tuple[BackendArg, ...], dict[str, BackendArg]]: Normalized `(args, kwargs)` pair.
    """
    from collections.abc import Sized

    s_val = args[1] if len(args) >= 2 else kwargs.get("s")
    axes_val = args[2] if len(args) >= 3 else kwargs.get("axes")
    return (
        (args[:2], {**kwargs, "axes": tuple(range(-len(s_val), 0))})
        if s_val is not None and axes_val is None and isinstance(s_val, Sized) and not isinstance(s_val, (str, bytes))
        else (args, kwargs)
    )


@global_eager_registry.register("Fftn")
def _np_fftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fftn operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    norm_args, norm_kwargs = _with_default_fft_axes(args, kwargs)
    return adapt_backend_module(backend_module).require_symbol("fft").fftn(*norm_args, **norm_kwargs)


@global_eager_registry.register("Ifftn")
def _np_ifftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifftn operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    norm_args, norm_kwargs = _with_default_fft_axes(args, kwargs)
    return adapt_backend_module(backend_module).require_symbol("fft").ifftn(*norm_args, **norm_kwargs)


@global_eager_registry.register("Rfftn")
def _np_rfftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfftn operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    norm_args, norm_kwargs = _with_default_fft_axes(args, kwargs)
    return adapt_backend_module(backend_module).require_symbol("fft").rfftn(*norm_args, **norm_kwargs)


@global_eager_registry.register("Irfftn")
def _np_irfftn(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np irfftn operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    norm_args, norm_kwargs = _with_default_fft_axes(args, kwargs)
    return adapt_backend_module(backend_module).require_symbol("fft").irfftn(*norm_args, **norm_kwargs)


@global_eager_registry.register("Fft2")
def _np_fft2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fft2 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").fft2(*args, **kwargs)


@global_eager_registry.register("Ifft2")
def _np_ifft2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifft2 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").ifft2(*args, **kwargs)


@global_eager_registry.register("Rfft2")
def _np_rfft2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfft2 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").rfft2(*args, **kwargs)


@global_eager_registry.register("Irfft2")
def _np_irfft2(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np irfft2 operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").irfft2(*args, **kwargs)


@global_eager_registry.register("Fftnd")
def _np_fftnd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fftnd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    norm_args, norm_kwargs = _with_default_fft_axes(args, kwargs)
    return adapt_backend_module(backend_module).require_symbol("fft").fftn(*norm_args, **norm_kwargs)


@global_eager_registry.register("Ifftnd")
def _np_ifftnd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifftnd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    norm_args, norm_kwargs = _with_default_fft_axes(args, kwargs)
    return adapt_backend_module(backend_module).require_symbol("fft").ifftn(*norm_args, **norm_kwargs)


@global_eager_registry.register("Rfftnd")
def _np_rfftnd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfftnd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    norm_args, norm_kwargs = _with_default_fft_axes(args, kwargs)
    return adapt_backend_module(backend_module).require_symbol("fft").rfftn(*norm_args, **norm_kwargs)


@global_eager_registry.register("Irfftnd")
def _np_irfftnd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np irfftnd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    norm_args, norm_kwargs = _with_default_fft_axes(args, kwargs)
    return adapt_backend_module(backend_module).require_symbol("fft").irfftn(*norm_args, **norm_kwargs)


@global_eager_registry.register("Fftshift")
def _np_fftshift(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fftshift operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").fftshift(*args, **kwargs)


@global_eager_registry.register("Ifftshift")
def _np_ifftshift(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ifftshift operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").ifftshift(*args, **kwargs)


@global_eager_registry.register("Fftfreq")
def _np_fftfreq(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fftfreq operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").fftfreq(*args, **kwargs)


@global_eager_registry.register("Hfft")
def _np_hfft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np hfft operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").hfft(*args, **kwargs)


@global_eager_registry.register("Ihfft")
def _np_ihfft(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np ihfft operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").ihfft(*args, **kwargs)


@global_eager_registry.register("Rfftfreq")
def _np_rfftfreq(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np rfftfreq operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fft").rfftfreq(*args, **kwargs)
