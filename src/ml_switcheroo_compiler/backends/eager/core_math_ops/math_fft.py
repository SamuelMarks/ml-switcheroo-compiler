# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Fft")
def _fft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Rfft")
def _rfft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _rfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.rfft(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fftn")
def _fftn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    fft_mod = getattr(backend_module, "fft", None)
    return fft_mod.fftn(*args, **kwargs) if fft_mod else None


@global_eager_registry.register("Fft2")
def _fft2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "fft2"):
        return func.fft2(*args, **kwargs)

    x = args[0]
    return backend_module.fft.fft2(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Fftfreq")
def _fftfreq(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftfreq operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "fftfreq"):
        return func.fftfreq(*args, **kwargs)

    n = args[0]
    d = kwargs.get("d", 1.0)
    return backend_module.fft.fftfreq(n, d=d)


@global_eager_registry.register("Fftnd")
def _fftnd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "fftn"):
        return func.fftn(*args, **kwargs)

    x = args[0]
    return backend_module.fft.fftn(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Fftshift")
def _fftshift(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftshift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "fftshift"):
        return func.fftshift(*args, **kwargs)

    x = args[0]
    return backend_module.fft.fftshift(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Ifft")
def _ifft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ifft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "ifft"):
        return func.ifft(*args, **kwargs)

    x = args[0]
    return backend_module.fft.ifft(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Ifft2")
def _ifft2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ifft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "ifft2"):
        return func.ifft2(*args, **kwargs)

    x = args[0]
    return backend_module.fft.ifft2(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Ifftn")
def _ifftn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ifftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "ifftn"):
        return func.ifftn(*args, **kwargs)

    x = args[0]
    return backend_module.fft.ifftn(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Ifftshift")
def _ifftshift(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ifftshift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fft", None)
    if func and hasattr(func, "ifftshift"):
        return func.ifftshift(*args, **kwargs)

    x = args[0]
    return backend_module.fft.ifftshift(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Fftconvolve")
def _fftconvolve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fftconvolve operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.signal

    func = getattr(backend_module, "fftconvolve", None)
    if func:
        return func(*args, **kwargs)

    x, y = args[0], args[1]
    return scipy.signal.fftconvolve(backend_module.asarray(x), backend_module.asarray(y), **kwargs)


@global_eager_registry.register("Hfft")
def _np_hfft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_hfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "hfft", getattr(backend_module, "hfft", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.fft.hfft(*args, **kwargs)
