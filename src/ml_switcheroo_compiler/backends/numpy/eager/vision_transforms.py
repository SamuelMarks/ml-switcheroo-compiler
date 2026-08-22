# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy vision transforms."""

from typing import Any

from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    random_elastic_transform_eager,
    random_perspective_eager,
    random_shear_eager,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("RandomShear")
def _np_random_shear(backend_module: Any, images: Any, y_factor: Any, x_factor: Any = None, **kwargs: Any) -> Any:
    """Evaluate _np_random_shear operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        y_factor (object): The y_factor parameter.
        x_factor (object): The x_factor parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return random_shear_eager(backend_module, images, y_factor, x_factor, **kwargs)


@numpy_eager_registry.register("RandomPerspective")
def _np_random_perspective(backend_module: Any, images: Any, factor: Any, **kwargs: Any) -> Any:
    """Evaluate _np_random_perspective operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        factor (object): The factor parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return random_perspective_eager(backend_module, images, factor, **kwargs)


@numpy_eager_registry.register("RandomElasticTransform")
def _np_random_elastic_transform(backend_module: Any, images: Any, alpha: Any, sigma: Any, **kwargs: Any) -> Any:
    """Evaluate _np_random_elastic_transform operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        alpha (object): The alpha parameter.
        sigma (object): The sigma parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return random_elastic_transform_eager(backend_module, images, alpha, sigma, **kwargs)
