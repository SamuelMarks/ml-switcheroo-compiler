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
    """Evaluate _np_random_shear operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        y_factor (object): The y_factor parameter.
        x_factor (object): The x_factor parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return random_shear_eager(backend_module, images, y_factor, x_factor, **kwargs)


@numpy_eager_registry.register("RandomPerspective")
def _np_random_perspective(backend_module: object, images: object, factor: object, **kwargs: object) -> object:
    """Evaluate _np_random_perspective operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        factor (object): The factor parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return random_perspective_eager(backend_module, images, factor, **kwargs)


@numpy_eager_registry.register("RandomElasticTransform")
def _np_random_elastic_transform(backend_module: object, images: object, alpha: object, sigma: object, **kwargs: object) -> object:
    """Evaluate _np_random_elastic_transform operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        alpha (object): The alpha parameter.
        sigma (object): The sigma parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return random_elastic_transform_eager(backend_module, images, alpha, sigma, **kwargs)
