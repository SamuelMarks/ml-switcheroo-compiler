# ruff: noqa: F405, F403
"""Shared vision utilities and ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Degeneration")
def _np_degeneration(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("GaussianBlur")
def _np_gaussian_blur(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.eager.signal import gaussian_blur_eager

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        from ml_switcheroo_compiler.ops.configs import BlurConfig

        config_obj = BlurConfig(
            kernel_size=config_obj.get("kernel_size", (3, 3)),
            sigma=config_obj.get("sigma", (1.0, 1.0)),
            data_format=config_obj.get("data_format", None),
        )

    return gaussian_blur_eager(backend_module, images, config_obj)


@numpy_eager_registry.register("MedianFilter")
def _np_median_filter(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.eager import median_filter_eager

    return median_filter_eager(backend_module, images, **kwargs)


@numpy_eager_registry.register("Sharpen")
def _np_sharpen(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
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
