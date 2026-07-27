# ruff: noqa: E501
"""Numpy vision transforms."""

from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    random_elastic_transform_eager,
    random_perspective_eager,
    random_shear_eager,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("RandomShear")
def _np_random_shear(backend_module: object, images: object, y_factor: object, x_factor: object = None, **kwargs: object) -> object:
    """Evaluate the random shear logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        images (object): Required parameter for images.
        y_factor (object): Required parameter for y_factor.
        x_factor (object): Required parameter for x_factor.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return random_shear_eager(backend_module, images, y_factor, x_factor, **kwargs)


@numpy_eager_registry.register("RandomPerspective")
def _np_random_perspective(backend_module: object, images: object, factor: object, **kwargs: object) -> object:
    """Evaluate the random perspective logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        images (object): Required parameter for images.
        factor (object): Required parameter for factor.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return random_perspective_eager(backend_module, images, factor, **kwargs)


@numpy_eager_registry.register("RandomElasticTransform")
def _np_random_elastic_transform(backend_module: object, images: object, alpha: object, sigma: object, **kwargs: object) -> object:
    """Evaluate the random elastic transform logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        images (object): Required parameter for images.
        alpha (object): Required parameter for alpha.
        sigma (object): Required parameter for sigma.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return random_elastic_transform_eager(backend_module, images, alpha, sigma, **kwargs)
