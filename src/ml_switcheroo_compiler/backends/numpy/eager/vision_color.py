# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Shared vision utilities and ops."""

from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager.utils import _from_channels_last, _to_channels_last
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AdjustBrightness")
def _np_adjust_brightness(backend_module: Any, images: Any, delta: float, **kwargs: Any) -> Any:
    """Evaluate _np_adjust_brightness operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        delta (float): The delta parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return np.clip(images + delta, 0.0, 1.0)


@numpy_eager_registry.register("AdjustContrast")
def _np_adjust_contrast(backend_module: Any, images: Any, contrast_factor: float, **kwargs: Any) -> Any:
    """Evaluate _np_adjust_contrast operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        contrast_factor (float): The contrast_factor parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    mean = np.mean(images, axis=(-3, -2), keepdims=True)
    return np.clip((images - mean) * contrast_factor + mean, 0.0, 1.0)


@numpy_eager_registry.register("AdjustHue")
def _np_adjust_hue(backend_module: Any, images: Any, delta: float, **kwargs: Any) -> Any:
    """Evaluate _np_adjust_hue operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        delta (float): The delta parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return images


@numpy_eager_registry.register("AdjustSaturation")
def _np_adjust_saturation(backend_module: Any, images: Any, saturation_factor: float, **kwargs: Any) -> Any:
    """Evaluate _np_adjust_saturation operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        saturation_factor (float): The saturation_factor parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    gray = _np_rgb_to_grayscale(backend_module, images)
    return np.clip(gray + (images - gray) * saturation_factor, 0.0, 1.0)


@numpy_eager_registry.register("AutoContrast")
def _np_auto_contrast(backend_module: Any, images: Any, **kwargs: Any) -> Any:
    """Evaluate _np_auto_contrast operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    value_range = kwargs.get("value_range", (0, 255))
    low = np.min(images, axis=(-3, -2), keepdims=True)
    high = np.max(images, axis=(-3, -2), keepdims=True)
    diff = high - low
    diff = np.where(diff == 0.0, 1.0, diff)
    out = (images - low) / diff
    return np.clip(out * (value_range[1] - value_range[0]) + value_range[0], value_range[0], value_range[1]).astype(images.dtype)


@numpy_eager_registry.register("Equalization")
def _np_equalization(backend_module: Any, images: Any, **kwargs: Any) -> Any:
    """Evaluate _np_equalization operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    images_uint8 = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    out = np.empty_like(images_uint8)
    for b in range(images.shape[0]):
        for c in range(images.shape[-1]):
            (hist, _) = np.histogram(images_uint8[b, ..., c].flatten(), 256, [0, 256])  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)  # type: ignore
            if cdf_m.max() - cdf_m.min() == 0:
                out[b, ..., c] = images_uint8[b, ..., c]
            else:
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf = np.ma.filled(cdf_m, 0).astype("uint8")  # type: ignore
                out[b, ..., c] = cdf[images_uint8[b, ..., c]]
    return out.astype(images.dtype) / 255.0


@numpy_eager_registry.register("Invert")
def _np_invert(backend_module: Any, images: Any, **kwargs: Any) -> Any:
    """Evaluate _np_invert operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    value_range = kwargs.get("value_range", (0, 255))
    return value_range[1] - images + value_range[0]


@numpy_eager_registry.register("Posterize")
def _np_posterize(backend_module: Any, images: Any, **kwargs: Any) -> Any:
    """Evaluate _np_posterize operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    bits = kwargs.get("bits", 4)
    shift = 8 - bits
    images_uint8 = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    posterized = np.bitwise_and(images_uint8, np.array(~((1 << shift) - 1) & 255, dtype=np.uint8))
    return posterized.astype(images.dtype) / 255.0


@numpy_eager_registry.register("RgbToGrayscale")
def _np_rgb_to_grayscale(backend_module: Any, images: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rgb_to_grayscale operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    np_mod = np
    data_format = kwargs.get("data_format", "channels_last")
    gray = _to_channels_last(np_mod, images, data_format)
    weights = np_mod.array([0.2989, 0.587, 0.114], dtype=gray.dtype)
    gray = np_mod.sum(gray * weights, axis=-1, keepdims=True)
    gray = _from_channels_last(np_mod, gray, data_format)
    return gray


@numpy_eager_registry.register("Solarize")
def _np_solarize(backend_module: Any, images: Any, **kwargs: Any) -> Any:
    """Evaluate _np_solarize operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    threshold = kwargs.get("threshold", 0.5)
    value_range = kwargs.get("value_range", (0, 255))
    return np.where(images >= threshold, value_range[1] - images + value_range[0], images)


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_np_adjust_brightness",
    "_np_adjust_contrast",
    "_np_adjust_hue",
    "_np_adjust_saturation",
    "_np_auto_contrast",
    "_np_equalization",
    "_np_invert",
    "_np_posterize",
    "_np_rgb_to_grayscale",
    "_np_solarize",
    "np",
    "numpy_eager_registry",
]
