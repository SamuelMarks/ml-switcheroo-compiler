"""Extracted math functions for numpy eager."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

# pragma: no cover


@numpy_eager_registry.register("Fft")
def _np_fft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.fft(*args, **kwargs)


@numpy_eager_registry.register("Rfft")
def _np_rfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.rfft(*args, **kwargs)


@numpy_eager_registry.register("Ifft")
def _np_ifft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.ifft(*args, **kwargs)


@numpy_eager_registry.register("Irfft")
def _np_irfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.irfft(*args, **kwargs)


@numpy_eager_registry.register("Fftn")
def _np_fftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.fftn(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Ifftn")
def _np_ifftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.ifftn(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Rfftn")
def _np_rfftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.rfftn(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Irfftn")
def _np_irfftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.irfftn(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Fft2")
def _np_fft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.fft2(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Ifft2")
def _np_ifft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.ifft2(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Rfft2")
def _np_rfft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.rfft2(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Irfft2")
def _np_irfft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.fft.irfft2(*args, **kwargs)  # pragma: no cover
