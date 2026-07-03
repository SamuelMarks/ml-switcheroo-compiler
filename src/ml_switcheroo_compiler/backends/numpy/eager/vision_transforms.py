"""Numpy vision transforms."""

from ml_switcheroo_compiler.backends.eager.vision_geometric import (  # pragma: no cover
    random_elastic_transform_eager,
    random_perspective_eager,
    random_shear_eager,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("RandomShear")
def _np_random_shear(
    backend_module: object,
    images: object,
    y_factor: object,
    x_factor: object = None,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        y_factor: Arg.
        x_factor: Arg.
        kwargs: Arg.
    """
    return random_shear_eager(  # pragma: no cover
        backend_module,
        images,
        y_factor,
        x_factor,
        **kwargs,
    )


@numpy_eager_registry.register("RandomPerspective")
def _np_random_perspective(
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
    return random_perspective_eager(  # pragma: no cover
        backend_module,
        images,
        factor,
        **kwargs,
    )


@numpy_eager_registry.register("RandomElasticTransform")
def _np_random_elastic_transform(
    backend_module: object,
    images: object,
    alpha: object,
    sigma: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        alpha: Arg.
        sigma: Arg.
        kwargs: Arg.
    """
    return random_elastic_transform_eager(  # pragma: no cover
        backend_module,
        images,
        alpha,
        sigma,
        **kwargs,
    )
