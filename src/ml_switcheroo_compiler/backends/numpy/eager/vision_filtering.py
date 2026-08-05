# ruff: noqa: E501
"""Shared vision utilities and ops."""

from ml_switcheroo_compiler.backends.eager import median_filter_eager
from ml_switcheroo_compiler.backends.eager.signal import gaussian_blur_eager
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.ops.configs import BlurConfig


@numpy_eager_registry.register("Degeneration")
def _np_degeneration(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_degeneration operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return images


@numpy_eager_registry.register("GaussianBlur")
def _np_gaussian_blur(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_gaussian_blur operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        config_obj = BlurConfig(
            kernel_size=config_obj.get("kernel_size", (3, 3)),
            sigma=config_obj.get("sigma", (1.0, 1.0)),
            data_format=config_obj.get("data_format", None),
        )
    return gaussian_blur_eager(backend_module, images, config_obj)


@numpy_eager_registry.register("MedianFilter")
def _np_median_filter(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_median_filter operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return median_filter_eager(backend_module, images, **kwargs)


@numpy_eager_registry.register("Sharpen")
def _np_sharpen(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_sharpen operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return images


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_np_degeneration",
    "_np_gaussian_blur",
    "_np_median_filter",
    "_np_sharpen",
    "numpy_eager_registry",
]
