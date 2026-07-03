"""Numpy vision filters."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_extras import _np_gaussian_blur, _np_sharpen


@numpy_eager_registry.register("RandomGaussianBlur")
def _np_random_gaussian_blur(
    backend_module: object,
    images: object,
    kernel_size: object,
    sigma: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kernel_size: Arg.
        sigma: Arg.
        kwargs: Arg.
    """
    return _np_gaussian_blur(backend_module, images, kernel_size=kernel_size, sigma=sigma, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("RandomSharpness")
def _np_random_sharpness(
    backend_module: object,
    images: object,
    factor: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        factor: Arg.
        kwargs: Arg.
    """
    return _np_sharpen(backend_module, images, factor=factor, **kwargs)  # pragma: no cover
