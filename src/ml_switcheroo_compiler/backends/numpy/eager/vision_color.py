"""Shared vision utilities and color transformation operations."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.utils import _from_channels_last, _to_channels_last
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_float, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member


@numpy_eager_registry.register("AdjustBrightness")
def _np_adjust_brightness(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    delta: float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Adjust the brightness of images by delta.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        delta (float): Amount to add to pixel values.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Brightness adjusted images clipped to [0, 1].
    """
    return ensure_ndarray(np.clip(np.add(ensure_ndarray(images), delta), 0.0, 1.0))


@numpy_eager_registry.register("AdjustContrast")
def _np_adjust_contrast(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    contrast_factor: float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Adjust the contrast of images.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        contrast_factor (float): Multiplier for image contrast.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Contrast adjusted images clipped to [0, 1].
    """
    img_arr = ensure_ndarray(images)
    mean = ensure_ndarray(get_backend_member(np, "mean")(img_arr, axis=(-3, -2), keepdims=True))
    return ensure_ndarray(np.clip((img_arr - mean) * contrast_factor + mean, 0.0, 1.0))


@numpy_eager_registry.register("AdjustHue")
def _np_adjust_hue(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    delta: float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Adjust the hue of RGB images by delta.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        delta (float): Hue adjustment amount.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Hue adjusted images.
    """
    return ensure_ndarray(images)


@numpy_eager_registry.register("AdjustSaturation")
def _np_adjust_saturation(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    saturation_factor: float,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Adjust the saturation of RGB images.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        saturation_factor (float): Multiplier for color saturation.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Saturation adjusted images clipped to [0, 1].
    """
    img_arr = ensure_ndarray(images)
    gray = np.asarray(_np_rgb_to_grayscale(backend_module, img_arr))
    return ensure_ndarray(np.clip(np.add(gray, np.subtract(img_arr, gray) * saturation_factor), 0.0, 1.0))


@numpy_eager_registry.register("AutoContrast")
def _np_auto_contrast(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    **kwargs: tuple[int, int] | str | float | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Maximize the contrast of an image by stretching range.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        **kwargs (Union[tuple[int, int], str, float, None]): Keyword arguments such as value_range.

    Returns:
        np.ndarray: Contrast normalized images.
    """
    img_arr = ensure_ndarray(images)
    value_range_val = kwargs.get("value_range", (0, 255))
    value_range = value_range_val if isinstance(value_range_val, tuple) else (0, 255)
    low = ensure_ndarray(get_backend_member(np, "min")(img_arr, axis=(-3, -2), keepdims=True))
    high = ensure_ndarray(get_backend_member(np, "max")(img_arr, axis=(-3, -2), keepdims=True))
    diff: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = high - low
    diff = np.where(diff == 0.0, 1.0, diff)
    out = (img_arr - low) / diff
    return ensure_ndarray(
        np.clip(
            out * (value_range[1] - value_range[0]) + value_range[0],
            value_range[0],
            value_range[1],
        ).astype(img_arr.dtype)
    )


@numpy_eager_registry.register("Equalization")
def _np_equalization(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    **kwargs: float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Equalize image histogram across channels.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array in range [0, 1].
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        np.ndarray: Equalized images array.
    """
    img_arr = ensure_ndarray(images)
    images_uint8 = np.clip(np.multiply(img_arr, 255.0), 0, 255).astype(np.uint8)
    out = np.empty_like(images_uint8)
    for b in range(img_arr.shape[0]):
        for c in range(img_arr.shape[-1]):
            (hist, _) = get_backend_member(np, "histogram")(images_uint8[b, ..., c].flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = get_backend_member(np.ma, "masked_equal")(cdf, 0)
            if cdf_m.max() - cdf_m.min() == 0:
                out[b, ..., c] = images_uint8[b, ..., c]
            else:
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf_filled = get_backend_member(np.ma, "filled")(cdf_m, 0).astype("uint8")
                out[b, ..., c] = cdf_filled[images_uint8[b, ..., c]]
    return ensure_ndarray(np.divide(out.astype(img_arr.dtype), 255.0))


@numpy_eager_registry.register("Invert")
def _np_invert(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    **kwargs: tuple[int, int] | float | int | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Invert pixel values within specified range.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        **kwargs (Union[tuple[int, int], float, int, str, None]): Keyword arguments such as value_range.

    Returns:
        np.ndarray: Inverted images array.
    """
    img_arr = ensure_ndarray(images)
    value_range_val = kwargs.get("value_range", (0, 255))
    value_range = value_range_val if isinstance(value_range_val, tuple) else (0, 255)
    return ensure_ndarray(np.add(np.subtract(value_range[1], img_arr), value_range[0]))


@numpy_eager_registry.register("Posterize")
def _np_posterize(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    **kwargs: int | float | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Reduce the number of bits for each color channel.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        **kwargs (Union[int, float, str, None]): Keyword arguments such as bits.

    Returns:
        np.ndarray: Posterized images array.
    """
    img_arr = ensure_ndarray(images)
    bits_val = kwargs.get("bits", 4)
    bits = int(bits_val) if bits_val is not None else 4
    shift = 8 - bits
    images_uint8 = np.clip(np.multiply(img_arr, 255.0), 0, 255).astype(np.uint8)
    posterized = np.bitwise_and(images_uint8, np.array(~((1 << shift) - 1) & 255, dtype=np.uint8))
    return ensure_ndarray(np.divide(posterized.astype(img_arr.dtype), 255.0))


@numpy_eager_registry.register("RgbToGrayscale")
def _np_rgb_to_grayscale(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    **kwargs: str | float | int | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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
    gray = ensure_ndarray(_to_channels_last(np, images, data_format))
    weights = np.array([0.2989, 0.587, 0.114], dtype=get_backend_member(gray, "dtype"))
    gray = np.sum(gray * weights, axis=-1, keepdims=True)
    res = _from_channels_last(np, gray, data_format)
    return ensure_ndarray(np.asarray(res))


@numpy_eager_registry.register("Solarize")
def _np_solarize(
    backend_module: BackendModuleProtocol,
    images: EagerValue,
    **kwargs: float | tuple[int, int] | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Invert pixel values above a threshold.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        **kwargs (Union[float, tuple[int, int], str, None]): Keyword arguments (threshold, value_range).

    Returns:
        np.ndarray: Solarized images array.
    """
    img_arr = ensure_ndarray(images)
    threshold_val = kwargs.get("threshold", 0.5)
    threshold = float(ensure_float(threshold_val)) if threshold_val is not None else 0.5
    value_range_val = kwargs.get("value_range", (0, 255))
    value_range = value_range_val if isinstance(value_range_val, tuple) else (0, 255)
    return ensure_ndarray(np.where(np.greater_equal(img_arr, threshold), np.add(np.subtract(value_range[1], img_arr), value_range[0]), img_arr))


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
