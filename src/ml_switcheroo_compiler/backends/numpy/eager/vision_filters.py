# ruff: noqa: E501
"""Numpy vision filters."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import _np_gaussian_blur, _np_sharpen


@numpy_eager_registry.register("RandomGaussianBlur")
def _np_random_gaussian_blur(backend_module: object, images: object, kernel_size: object, sigma: object, **kwargs: object) -> object:
    """Evaluate the random gaussian blur logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        images (object): Required parameter for images.
        kernel_size (object): Required parameter for kernel_size.
        sigma (object): Required parameter for sigma.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _np_gaussian_blur(backend_module, images, kernel_size=kernel_size, sigma=sigma, **kwargs)


@numpy_eager_registry.register("RandomSharpness")
def _np_random_sharpness(backend_module: object, images: object, factor: object, **kwargs: object) -> object:
    """Evaluate the random sharpness logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        images (object): Required parameter for images.
        factor (object): Required parameter for factor.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _np_sharpen(backend_module, images, factor=factor, **kwargs)
