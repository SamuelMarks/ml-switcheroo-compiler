# ruff: noqa: E501
"""Numpy vision filters."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import _np_gaussian_blur, _np_sharpen


@numpy_eager_registry.register("RandomGaussianBlur")
def _np_random_gaussian_blur(backend_module: object, images: object, kernel_size: object, sigma: object, **kwargs: object) -> object:
    """Evaluate _np_random_gaussian_blur operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        kernel_size (object): The kernel_size parameter.
        sigma (object): The sigma parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _np_gaussian_blur(backend_module, images, kernel_size=kernel_size, sigma=sigma, **kwargs)


@numpy_eager_registry.register("RandomSharpness")
def _np_random_sharpness(backend_module: object, images: object, factor: object, **kwargs: object) -> object:
    """Evaluate _np_random_sharpness operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        factor (object): The factor parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return _np_sharpen(backend_module, images, factor=factor, **kwargs)
