"""Shared vision utilities and color transformation operations."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.utils import _from_channels_last, _to_channels_last
from ml_switcheroo_compiler.backends.eager_registry import as_float, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("AdjustBrightness")
def _np_adjust_brightness(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    delta: float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Adjust the brightness of images by delta.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        delta (float): Amount to add to pixel values.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Brightness adjusted images clipped to [0, 1].
    """
    return np.clip(np.add(images, delta), 0.0, 1.0)


@numpy_eager_registry.register("AdjustContrast")
def _np_adjust_contrast(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    contrast_factor: float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Adjust the contrast of images.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        contrast_factor (float): Multiplier for image contrast.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Contrast adjusted images clipped to [0, 1].
    """
    mean = getattr(np, "mean")(images, axis=(-3, -2), keepdims=True)
    return as_ndarray(np.clip((images - mean) * contrast_factor + mean, 0.0, 1.0))


@numpy_eager_registry.register("AdjustHue")
def _np_adjust_hue(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    delta: float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Adjust the hue of RGB images by delta.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        delta (float): Hue adjustment amount.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Hue adjusted images.
    """
    return images


@numpy_eager_registry.register("AdjustSaturation")
def _np_adjust_saturation(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    saturation_factor: float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Adjust the saturation of RGB images.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        saturation_factor (float): Multiplier for color saturation.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Saturation adjusted images clipped to [0, 1].
    """
    gray = np.asarray(_np_rgb_to_grayscale(backend_module, images))
    return np.asarray(np.clip(np.add(gray, np.subtract(images, gray) * saturation_factor), 0.0, 1.0))


@numpy_eager_registry.register("AutoContrast")
def _np_auto_contrast(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: tuple[int, int] | str | float | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Maximize the contrast of an image by stretching range.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        **kwargs (Union[tuple[int, int], str, float, None]): Keyword arguments such as value_range.

    Returns:
        np.ndarray: Contrast normalized images.
    """
    value_range_val = kwargs.get("value_range", (0, 255))
    value_range = value_range_val if isinstance(value_range_val, tuple) else (0, 255)
    low = np.min(images, axis=(-3, -2), keepdims=True)
    high = np.max(images, axis=(-3, -2), keepdims=True)
    diff = high - low
    diff = np.where(diff == 0.0, 1.0, diff)
    out = (images - low) / diff
    return as_ndarray(
        np.clip(
            out * (value_range[1] - value_range[0]) + value_range[0],
            value_range[0],
            value_range[1],
        ).astype(images.dtype)
    )


@numpy_eager_registry.register("Equalization")
def _np_equalization(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Equalize image histogram across channels.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array in range [0, 1].
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Equalized images array.
    """
    images_uint8 = np.clip(np.multiply(images, 255.0), 0, 255).astype(np.uint8)
    out = np.empty_like(images_uint8)
    for b in range(images.shape[0]):
        for c in range(images.shape[-1]):
            (hist, _) = getattr(np, "histogram")(images_uint8[b, ..., c].flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = getattr(np.ma, "masked_equal")(cdf, 0)
            if cdf_m.max() - cdf_m.min() == 0:
                out[b, ..., c] = images_uint8[b, ..., c]
            else:
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf_filled = getattr(np.ma, "filled")(cdf_m, 0).astype("uint8")
                out[b, ..., c] = cdf_filled[images_uint8[b, ..., c]]
    return np.divide(out.astype(images.dtype), 255.0)


@numpy_eager_registry.register("Invert")
def _np_invert(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: tuple[int, int] | float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Invert pixel values within specified range.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        **kwargs (Union[tuple[int, int], float, int, str, None]): Keyword arguments such as value_range.

    Returns:
        np.ndarray: Inverted images array.
    """
    value_range_val = kwargs.get("value_range", (0, 255))
    value_range = value_range_val if isinstance(value_range_val, tuple) else (0, 255)
    return np.add(np.subtract(value_range[1], images), value_range[0])


@numpy_eager_registry.register("Posterize")
def _np_posterize(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: int | float | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Reduce the number of bits for each color channel.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        **kwargs (Union[int, float, str, None]): Keyword arguments such as bits.

    Returns:
        np.ndarray: Posterized images array.
    """
    bits_val = kwargs.get("bits", 4)
    bits = int(bits_val) if bits_val is not None else 4
    shift = 8 - bits
    images_uint8 = np.clip(np.multiply(images, 255.0), 0, 255).astype(np.uint8)
    posterized = np.bitwise_and(images_uint8, np.array(~((1 << shift) - 1) & 255, dtype=np.uint8))
    return np.divide(posterized.astype(images.dtype), 255.0)


@numpy_eager_registry.register("RgbToGrayscale")
def _np_rgb_to_grayscale(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: str | float | int | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Convert RGB images to grayscale.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        **kwargs (Union[str, float, int, None]): Keyword arguments such as data_format.

    Returns:
        np.ndarray: Grayscale images array.
    """
    data_format_val = kwargs.get("data_format", "channels_last")
    data_format = str(data_format_val) if data_format_val is not None else "channels_last"
    gray = _to_channels_last(np, images, data_format)
    weights = np.array([0.2989, 0.587, 0.114], dtype=getattr(gray, "dtype"))
    gray = np.sum(gray * weights, axis=-1, keepdims=True)
    res = _from_channels_last(np, gray, data_format)
    return np.asarray(res)


@numpy_eager_registry.register("Solarize")
def _np_solarize(
    backend_module: BackendModuleProtocol,
    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    **kwargs: float | tuple[int, int] | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Invert pixel values above a threshold.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        **kwargs (Union[float, tuple[int, int], str, None]): Keyword arguments (threshold, value_range).

    Returns:
        np.ndarray: Solarized images array.
    """
    threshold_val = kwargs.get("threshold", 0.5)
    threshold = float(as_float(threshold_val)) if threshold_val is not None else 0.5
    value_range_val = kwargs.get("value_range", (0, 255))
    value_range = value_range_val if isinstance(value_range_val, tuple) else (0, 255)
    return np.where(np.greater_equal(images, threshold), np.add(np.subtract(value_range[1], images), value_range[0]), images)


__all__ = [
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
]
