# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Shared vision utilities and ops."""

import numpy as np

from ml_switcheroo_compiler.backends.eager.utils import _from_channels_last, _to_channels_last
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AdjustBrightness")
def _np_adjust_brightness(backend_module: object, images: object, delta: float, **kwargs: object) -> object:
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
def _np_adjust_contrast(backend_module: object, images: object, contrast_factor: float, **kwargs: object) -> object:
    """Evaluate _np_adjust_contrast operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        contrast_factor (float): The contrast_factor parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    mean: object = np.mean(images, axis=(-3, -2), keepdims=True)
    return np.clip((images - mean) * contrast_factor + mean, 0.0, 1.0)


@numpy_eager_registry.register("AdjustHue")
def _np_adjust_hue(backend_module: object, images: object, delta: float, **kwargs: object) -> object:
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
def _np_adjust_saturation(backend_module: object, images: object, saturation_factor: float, **kwargs: object) -> object:
    """Evaluate _np_adjust_saturation operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        saturation_factor (float): The saturation_factor parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    gray: object = _np_rgb_to_grayscale(backend_module, images)
    return np.clip(gray + (images - gray) * saturation_factor, 0.0, 1.0)


@numpy_eager_registry.register("AutoContrast")
def _np_auto_contrast(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_auto_contrast operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    value_range: object = kwargs.get("value_range", (0, 255))
    low: object = np.min(images, axis=(-3, -2), keepdims=True)
    high: object = np.max(images, axis=(-3, -2), keepdims=True)
    diff: object = high - low
    diff: object = np.where(diff == 0.0, 1.0, diff)
    out: object = (images - low) / diff
    return np.clip(out * (value_range[1] - value_range[0]) + value_range[0], value_range[0], value_range[1]).astype(images.dtype)


@numpy_eager_registry.register("Equalization")
def _np_equalization(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_equalization operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    images_uint8: object = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    out: object = np.empty_like(images_uint8)
    for b in range(images.shape[0]):
        for c in range(images.shape[-1]):
            (hist, _) = np.histogram(images_uint8[b, ..., c].flatten(), 256, [0, 256])
            cdf: object = hist.cumsum()
            cdf_m: object = np.ma.masked_equal(cdf, 0)
            if cdf_m.max() - cdf_m.min() == 0:
                out[b, ..., c] = images_uint8[b, ..., c]
            else:
                cdf_m: object = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf: object = np.ma.filled(cdf_m, 0).astype("uint8")
                out[b, ..., c] = cdf[images_uint8[b, ..., c]]
    return out.astype(images.dtype) / 255.0


@numpy_eager_registry.register("Invert")
def _np_invert(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_invert operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    value_range: object = kwargs.get("value_range", (0, 255))
    return value_range[1] - images + value_range[0]


@numpy_eager_registry.register("Posterize")
def _np_posterize(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_posterize operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    bits: object = kwargs.get("bits", 4)
    shift: object = 8 - bits
    images_uint8: object = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    posterized: object = np.bitwise_and(images_uint8, np.array(~((1 << shift) - 1) & 255, dtype=np.uint8))
    return posterized.astype(images.dtype) / 255.0


@numpy_eager_registry.register("RgbToGrayscale")
def _np_rgb_to_grayscale(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_rgb_to_grayscale operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    np_mod: object = np
    data_format: object = kwargs.get("data_format", "channels_last")
    gray: object = _to_channels_last(np_mod, images, data_format)
    weights: object = np_mod.array([0.2989, 0.587, 0.114], dtype=gray.dtype)
    gray: object = np_mod.sum(gray * weights, axis=-1, keepdims=True)
    gray: object = _from_channels_last(np_mod, gray, data_format)
    return gray


@numpy_eager_registry.register("Solarize")
def _np_solarize(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_solarize operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    threshold: object = kwargs.get("threshold", 0.5)
    value_range: object = kwargs.get("value_range", (0, 255))
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
