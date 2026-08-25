# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Fft")
def _fft(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _fft operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.fft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Rfft")
def _rfft(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _rfft operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.rfft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fft2")
def _fft2(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _fft2 operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.fft2(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftconvolve")
def _fftconvolve(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _fftconvolve operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    if hasattr(fft_mod, "fftconvolve"):
        return fft_mod.fftconvolve(*args, **kwargs)
    elif hasattr(backend_module, "signal") and hasattr(backend_module.signal, "fftconvolve"):
        return backend_module.signal.fftconvolve(*args, **kwargs)
    return None


@global_eager_registry.register("Fftfreq")
def _fftfreq(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _fftfreq operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.fftfreq(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftn")
def _fftn(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _fftn operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftnd")
def _fftnd(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _fftnd operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    if hasattr(fft_mod, "fftnd"):
        return fft_mod.fftnd(*args, **kwargs)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftshift")
def _fftshift(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _fftshift operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.fftshift(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifft")
def _ifft(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _ifft operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.ifft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifft2")
def _ifft2(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _ifft2 operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.ifft2(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifftn")
def _ifftn(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _ifftn operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.ifftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Ifftshift")
def _ifftshift(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _ifftshift operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.ifftshift(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("NpHfft")
def _np_hfft(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _np_hfft operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    fft_mod: typing.Any = getattr(backend_module, "fft", None)
    return fft_mod.hfft(*args, **kwargs) if fft_mod else None
