"""Numpy vision filters."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import _np_gaussian_blur, _np_sharpen
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("RandomGaussianBlur")
def _np_random_gaussian_blur(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np random gaussian blur operation on the active backend array.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        kernel_size (tuple[int, int] | int): Blur kernel size.
        sigma (tuple[float, float] | float): Blur standard deviation.
        **kwargs (Union[float, int, str, None]): Keyword args.

    Returns:
        np.ndarray: Blurred images array.
    """
    return _np_gaussian_blur(backend_module, images, kernel_size=kernel_size, sigma=sigma, **kwargs)


@numpy_eager_registry.register("RandomSharpness")
def _np_random_sharpness(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    factor: float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np random sharpness operation on the active backend array.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        factor (float): Sharpness adjustment factor.
        **kwargs (Union[float, int, str, None]): Keyword args.

    Returns:
        np.ndarray: Sharpened images array.
    """
    return _np_sharpen(backend_module, images, factor=factor, **kwargs)
