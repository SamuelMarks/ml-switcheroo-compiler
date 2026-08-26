# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy vision filters."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import _np_gaussian_blur, _np_sharpen


@numpy_eager_registry.register("RandomGaussianBlur")
def _np_random_gaussian_blur(backend_module, images, kernel_size, sigma, **kwargs):
    """Evaluate _np_random_gaussian_blur operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        kernel_size (object): The kernel_size parameter.
        sigma (object): The sigma parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_gaussian_blur(backend_module, images, kernel_size=kernel_size, sigma=sigma, **kwargs)


@numpy_eager_registry.register("RandomSharpness")
def _np_random_sharpness(backend_module, images, factor, **kwargs):
    """Evaluate _np_random_sharpness operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        factor (object): The factor parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _np_sharpen(backend_module, images, factor=factor, **kwargs)
