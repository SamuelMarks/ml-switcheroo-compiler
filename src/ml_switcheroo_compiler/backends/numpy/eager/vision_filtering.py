"""Shared vision utilities and ops."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Callable, Optional, Union, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager import median_filter_eager
from ml_switcheroo_compiler.backends.eager.signal import gaussian_blur_eager
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, numpy_eager_registry

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.configs import BlurConfig


@numpy_eager_registry.register("Degeneration")
def _np_degeneration(
    backend_module: ModuleType,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np degeneration operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend module.
        images (np.ndarray): The images parameter.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    return images


@numpy_eager_registry.register("GaussianBlur")
def _np_gaussian_blur(
    backend_module: ModuleType,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: Union[BlurConfig, dict[str, Union[tuple[int, int], tuple[float, float], str, None]], float, int, str, bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np gaussian blur operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend module.
        images (np.ndarray): The images parameter.
        **kwargs (Union[BlurConfig, dict[str, Union[tuple[int, int], tuple[float, float], str, None]], float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    from ml_switcheroo_compiler.ops.configs import BlurConfig

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        config_obj = cast("Callable[..., BlurConfig]", BlurConfig)(
            kernel_size=config_obj.get("kernel_size", (3, 3)),
            sigma=config_obj.get("sigma", (1.0, 1.0)),
            data_format=cast("Optional[str]", config_obj.get("data_format", None)),
        )
    return np.asarray(cast("Callable[..., np.ndarray[tuple[int, ...], np.dtype[np.generic]]]", gaussian_blur_eager)(backend_module, images, config_obj))


@numpy_eager_registry.register("MedianFilter")
def _np_median_filter(
    backend_module: ModuleType,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np median filter operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend module.
        images (np.ndarray): The images parameter.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
    """
    return np.asarray(cast("Callable[..., EagerTensorProtocol]", median_filter_eager)(backend_module, images, **kwargs))


@numpy_eager_registry.register("Sharpen")
def _np_sharpen(
    backend_module: ModuleType,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager np sharpen operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend module.
        images (np.ndarray): The images parameter.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result.
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
