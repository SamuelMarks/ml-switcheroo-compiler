"""Numpy vision filters."""

from __future__ import annotations

from types import ModuleType
from typing import Union, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import _np_gaussian_blur, _np_sharpen


@numpy_eager_registry.register("RandomGaussianBlur")
def _np_random_gaussian_blur(
    backend_module: ModuleType,
    images: np.ndarray,
    kernel_size: tuple[int, int] | int,
    sigma: tuple[float, float] | float,
    **kwargs: Union[float, int, str, None],
) -> np.ndarray:
    """Evaluate _np_random_gaussian_blur operation.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        kernel_size (tuple[int, int] | int): Blur kernel size.
        sigma (tuple[float, float] | float): Blur standard deviation.
        **kwargs (Union[float, int, str, None]): Keyword args.

    Returns:
        np.ndarray: Blurred images array.
    """
    return cast(
        np.ndarray,
        _np_gaussian_blur(backend_module, images, kernel_size=kernel_size, sigma=sigma, **kwargs),
    )


@numpy_eager_registry.register("RandomSharpness")
def _np_random_sharpness(
    backend_module: ModuleType,
    images: np.ndarray,
    factor: float,
    **kwargs: Union[float, int, str, None],
) -> np.ndarray:
    """Evaluate _np_random_sharpness operation.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        factor (float): Sharpness adjustment factor.
        **kwargs (Union[float, int, str, None]): Keyword args.

    Returns:
        np.ndarray: Sharpened images array.
    """
    return cast(
        np.ndarray,
        _np_sharpen(backend_module, images, factor=factor, **kwargs),
    )
