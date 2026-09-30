"""Numpy vision transforms."""

from __future__ import annotations

from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    random_elastic_transform_eager,
    random_perspective_eager,
    random_shear_eager,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("RandomShear")
def _np_random_shear(
    backend_module: ModuleType,
    images: np.ndarray,
    y_factor: tuple[float, float] | float,
    x_factor: tuple[float, float] | float | None = None,
    **kwargs: Union[float, int, str, None],
) -> np.ndarray:
    """Evaluate _np_random_shear operation.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        y_factor (tuple[float, float] | float): Vertical shear factor.
        x_factor (tuple[float, float] | float | None): Horizontal shear factor.
        **kwargs (Union[float, int, str, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Sheared images array.
    """
    return np.asarray(random_shear_eager(backend_module, images, y_factor, x_factor, **kwargs))


@numpy_eager_registry.register("RandomPerspective")
def _np_random_perspective(
    backend_module: ModuleType,
    images: np.ndarray,
    factor: float,
    **kwargs: Union[float, int, str, None],
) -> np.ndarray:
    """Evaluate _np_random_perspective operation.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        factor (float): Perspective distortion factor.
        **kwargs (Union[float, int, str, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Transformed images array.
    """
    return np.asarray(random_perspective_eager(backend_module, images, factor, **kwargs))


@numpy_eager_registry.register("RandomElasticTransform")
def _np_random_elastic_transform(
    backend_module: ModuleType,
    images: np.ndarray,
    alpha: float,
    sigma: float,
    **kwargs: Union[float, int, str, None],
) -> np.ndarray:
    """Evaluate _np_random_elastic_transform operation.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        alpha (float): Scaling factor for displacement.
        sigma (float): Elastic filter smoothing factor.
        **kwargs (Union[float, int, str, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Elastically transformed images array.
    """
    return np.asarray(random_elastic_transform_eager(backend_module, images, alpha, sigma, **kwargs))
