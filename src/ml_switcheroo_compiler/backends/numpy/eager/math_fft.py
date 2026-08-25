# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Extracted math functions for numpy eager."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Fft")
def _np_fft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_fft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.fft(*args, **kwargs)


@numpy_eager_registry.register("Rfft")
def _np_rfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_rfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.rfft(*args, **kwargs)


@numpy_eager_registry.register("Ifft")
def _np_ifft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_ifft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.ifft(*args, **kwargs)


@numpy_eager_registry.register("Irfft")
def _np_irfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_irfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.irfft(*args, **kwargs)


@numpy_eager_registry.register("Fftn")
def _np_fftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_fftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.fftn(*args, **kwargs)


@numpy_eager_registry.register("Ifftn")
def _np_ifftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_ifftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.ifftn(*args, **kwargs)


@numpy_eager_registry.register("Rfftn")
def _np_rfftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_rfftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.rfftn(*args, **kwargs)


@numpy_eager_registry.register("Irfftn")
def _np_irfftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_irfftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.irfftn(*args, **kwargs)


@numpy_eager_registry.register("Fft2")
def _np_fft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_fft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.fft2(*args, **kwargs)


@numpy_eager_registry.register("Ifft2")
def _np_ifft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_ifft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.ifft2(*args, **kwargs)


@numpy_eager_registry.register("Rfft2")
def _np_rfft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_rfft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.rfft2(*args, **kwargs)


@numpy_eager_registry.register("Irfft2")
def _np_irfft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_irfft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.irfft2(*args, **kwargs)


@numpy_eager_registry.register("Fftnd")
def _np_fftnd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_fftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.fftn(*args, **kwargs)


@numpy_eager_registry.register("Ifftnd")
def _np_ifftnd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_ifftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.ifftn(*args, **kwargs)


@numpy_eager_registry.register("Rfftnd")
def _np_rfftnd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_rfftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.rfftn(*args, **kwargs)


@numpy_eager_registry.register("Irfftnd")
def _np_irfftnd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_irfftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.irfftn(*args, **kwargs)


@numpy_eager_registry.register("Fftshift")
def _np_fftshift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_fftshift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.fftshift(*args, **kwargs)


@numpy_eager_registry.register("Ifftshift")
def _np_ifftshift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_ifftshift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.ifftshift(*args, **kwargs)


@numpy_eager_registry.register("Fftfreq")
def _np_fftfreq(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_fftfreq operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.fftfreq(*args, **kwargs)


@numpy_eager_registry.register("Hfft")
def _np_hfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_hfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.hfft(*args, **kwargs)


@numpy_eager_registry.register("Ihfft")
def _np_ihfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_ihfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.ihfft(*args, **kwargs)


@numpy_eager_registry.register("Rfftfreq")
def _np_rfftfreq(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_rfftfreq operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fft.rfftfreq(*args, **kwargs)
