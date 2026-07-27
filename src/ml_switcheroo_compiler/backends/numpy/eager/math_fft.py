# ruff: noqa: E501
"""Extracted math functions for numpy eager."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Fft")
def _np_fft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fft logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.fft(*args, **kwargs)


@numpy_eager_registry.register("Rfft")
def _np_rfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rfft logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.rfft(*args, **kwargs)


@numpy_eager_registry.register("Ifft")
def _np_ifft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ifft logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.ifft(*args, **kwargs)


@numpy_eager_registry.register("Irfft")
def _np_irfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the irfft logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.irfft(*args, **kwargs)


@numpy_eager_registry.register("Fftn")
def _np_fftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fftn logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.fftn(*args, **kwargs)


@numpy_eager_registry.register("Ifftn")
def _np_ifftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ifftn logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.ifftn(*args, **kwargs)


@numpy_eager_registry.register("Rfftn")
def _np_rfftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rfftn logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.rfftn(*args, **kwargs)


@numpy_eager_registry.register("Irfftn")
def _np_irfftn(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the irfftn logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.irfftn(*args, **kwargs)


@numpy_eager_registry.register("Fft2")
def _np_fft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fft2 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.fft2(*args, **kwargs)


@numpy_eager_registry.register("Ifft2")
def _np_ifft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ifft2 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.ifft2(*args, **kwargs)


@numpy_eager_registry.register("Rfft2")
def _np_rfft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rfft2 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.rfft2(*args, **kwargs)


@numpy_eager_registry.register("Irfft2")
def _np_irfft2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the irfft2 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.irfft2(*args, **kwargs)


@numpy_eager_registry.register("Fftnd")
def _np_fftnd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fftnd logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.fftn(*args, **kwargs)


@numpy_eager_registry.register("Ifftnd")
def _np_ifftnd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ifftnd logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.ifftn(*args, **kwargs)


@numpy_eager_registry.register("Rfftnd")
def _np_rfftnd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rfftnd logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.rfftn(*args, **kwargs)


@numpy_eager_registry.register("Irfftnd")
def _np_irfftnd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the irfftnd logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.irfftn(*args, **kwargs)


@numpy_eager_registry.register("Fftshift")
def _np_fftshift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fftshift logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.fftshift(*args, **kwargs)


@numpy_eager_registry.register("Ifftshift")
def _np_ifftshift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ifftshift logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.ifftshift(*args, **kwargs)


@numpy_eager_registry.register("Fftfreq")
def _np_fftfreq(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the fftfreq logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.fftfreq(*args, **kwargs)


@numpy_eager_registry.register("Hfft")
def _np_hfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the hfft logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.hfft(*args, **kwargs)


@numpy_eager_registry.register("Ihfft")
def _np_ihfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the ihfft logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.ihfft(*args, **kwargs)


@numpy_eager_registry.register("Rfftfreq")
def _np_rfftfreq(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rfftfreq logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.fft.rfftfreq(*args, **kwargs)
