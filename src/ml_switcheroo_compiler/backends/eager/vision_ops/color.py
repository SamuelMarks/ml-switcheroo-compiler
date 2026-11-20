"""Generic backend-parametric eager reference operations for `vision_ops.color`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _from_channels_last, _resolve_eager_mod, _to_channels_last
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_float,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("AdjustBrightness")
def _np_adjust_brightness(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    delta: float,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Adjust the brightness of images by delta.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        delta (float): Amount to add to pixel values.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Brightness adjusted images clipped to [0, 1].
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return _ensure_arr(arr_mod.clip(arr_mod.add(_ensure_arr(images), delta), 0.0, 1.0))


@global_eager_registry.register("AdjustContrast")
def _np_adjust_contrast(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    contrast_factor: float,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Adjust the contrast of images.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        contrast_factor (float): Multiplier for image contrast.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Contrast adjusted images clipped to [0, 1].
    """
    arr_mod = _resolve_eager_mod(backend_module)
    img_arr = _ensure_arr(images)
    mean = _ensure_arr(adapt_backend_module(arr_mod).require_symbol("mean")(img_arr, axis=(-3, -2), keepdims=True))
    return _ensure_arr(arr_mod.clip((img_arr - mean) * contrast_factor + mean, 0.0, 1.0))


@global_eager_registry.register("AdjustHue")
def _np_adjust_hue(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    delta: float,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Adjust the hue of RGB images by delta.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        delta (float): Hue adjustment amount.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Hue adjusted images.
    """
    return _ensure_arr(images)


@global_eager_registry.register("AdjustSaturation")
def _np_adjust_saturation(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    saturation_factor: float,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Adjust the saturation of RGB images.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        saturation_factor (float): Multiplier for color saturation.
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Saturation adjusted images clipped to [0, 1].
    """
    arr_mod = _resolve_eager_mod(backend_module)
    img_arr = _ensure_arr(images)
    gray = arr_mod.asarray(_np_rgb_to_grayscale(backend_module, img_arr))
    return _ensure_arr(arr_mod.clip(arr_mod.add(gray, arr_mod.subtract(img_arr, gray) * saturation_factor), 0.0, 1.0))


@global_eager_registry.register("AutoContrast")
def _np_auto_contrast(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: tuple[int, int] | str | float | None,
) -> EagerTensorProtocol:
    """Maximize the contrast of an image by stretching range.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        **kwargs (Union[tuple[int, int], str, float, None]): Keyword arguments such as value_range.

    Returns:
        EagerTensorProtocol: Contrast normalized images.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    img_arr = _ensure_arr(images)
    value_range_val = kwargs.get("value_range", (0, 255))
    value_range = value_range_val if isinstance(value_range_val, tuple) else (0, 255)
    low = _ensure_arr(adapt_backend_module(arr_mod).require_symbol("min")(img_arr, axis=(-3, -2), keepdims=True))
    high = _ensure_arr(adapt_backend_module(arr_mod).require_symbol("max")(img_arr, axis=(-3, -2), keepdims=True))
    diff: EagerTensorProtocol = high - low
    diff = arr_mod.where(diff == 0.0, 1.0, diff)
    out = (img_arr - low) / diff
    return _ensure_arr(
        arr_mod.clip(
            out * (value_range[1] - value_range[0]) + value_range[0],
            value_range[0],
            value_range[1],
        ).astype(img_arr.dtype)
    )


@global_eager_registry.register("Equalization")
def _np_equalization(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: float | int | str | None,
) -> EagerTensorProtocol:
    """Equalize image histogram across channels.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array in range [0, 1].
        **kwargs (Union[float, int, str, None]): Keyword arguments.

    Returns:
        EagerTensorProtocol: Equalized images array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    img_arr = _ensure_arr(images)
    images_uint8 = arr_mod.clip(arr_mod.multiply(img_arr, 255.0), 0, 255).astype(arr_mod.uint8)
    out = arr_mod.empty_like(images_uint8)
    for b in range(img_arr.shape[0]):
        for c in range(img_arr.shape[-1]):
            (hist, _) = adapt_backend_module(arr_mod).require_symbol("histogram")(
                images_uint8[b, ..., c].flatten(), 256, [0, 256]
            )
            cdf = hist.cumsum()
            cdf_m = adapt_backend_module(adapt_backend_module(arr_mod).require_symbol("ma")).require_symbol(
                "masked_equal"
            )(cdf, 0)
            if cdf_m.max() - cdf_m.min() == 0:
                out[b, ..., c] = images_uint8[b, ..., c]
            else:
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf_filled = (
                    adapt_backend_module(adapt_backend_module(arr_mod).require_symbol("ma"))
                    .require_symbol("filled")(cdf_m, 0)
                    .astype("uint8")
                )
                out[b, ..., c] = cdf_filled[images_uint8[b, ..., c]]
    return _ensure_arr(arr_mod.divide(out.astype(img_arr.dtype), 255.0))


@global_eager_registry.register("Invert")
def _np_invert(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: tuple[int, int] | float | int | str | None,
) -> EagerTensorProtocol:
    """Invert pixel values within specified range.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        **kwargs (Union[tuple[int, int], float, int, str, None]): Keyword arguments such as value_range.

    Returns:
        EagerTensorProtocol: Inverted images array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    img_arr = _ensure_arr(images)
    value_range_val = kwargs.get("value_range", (0, 255))
    value_range = value_range_val if isinstance(value_range_val, tuple) else (0, 255)
    return _ensure_arr(arr_mod.add(arr_mod.subtract(value_range[1], img_arr), value_range[0]))


@global_eager_registry.register("Posterize")
def _np_posterize(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: int | float | str | None,
) -> EagerTensorProtocol:
    """Reduce the number of bits for each color channel.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        **kwargs (Union[int, float, str, None]): Keyword arguments such as bits.

    Returns:
        EagerTensorProtocol: Posterized images array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    img_arr = _ensure_arr(images)
    bits_val = kwargs.get("bits", 4)
    bits = int(bits_val) if bits_val is not None else 4
    shift = 8 - bits
    images_uint8 = arr_mod.clip(arr_mod.multiply(img_arr, 255.0), 0, 255).astype(arr_mod.uint8)
    posterized = arr_mod.bitwise_and(images_uint8, arr_mod.array(~((1 << shift) - 1) & 255, dtype=arr_mod.uint8))
    return _ensure_arr(arr_mod.divide(posterized.astype(img_arr.dtype), 255.0))


@global_eager_registry.register("RgbToGrayscale")
def _np_rgb_to_grayscale(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: str | float | int | None,
) -> EagerTensorProtocol:
    """Convert RGB images to grayscale.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        **kwargs (Union[str, float, int, None]): Keyword arguments such as data_format.

    Returns:
        EagerTensorProtocol: Grayscale images array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    data_format_val = kwargs.get("data_format", "channels_last")
    data_format = str(data_format_val) if data_format_val is not None else "channels_last"
    gray = _ensure_arr(_to_channels_last(arr_mod, images, data_format))
    weights = arr_mod.array([0.2989, 0.587, 0.114], dtype=adapt_backend_module(gray).require_symbol("dtype"))
    gray = arr_mod.sum(gray * weights, axis=-1, keepdims=True)
    res = _from_channels_last(arr_mod, gray, data_format)
    return _ensure_arr(arr_mod.asarray(res))


@global_eager_registry.register("Solarize")
def _np_solarize(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    **kwargs: float | tuple[int, int] | str | None,
) -> EagerTensorProtocol:
    """Invert pixel values above a threshold.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        **kwargs (Union[float, tuple[int, int], str, None]): Keyword arguments (threshold, value_range).

    Returns:
        EagerTensorProtocol: Solarized images array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    img_arr = _ensure_arr(images)
    threshold_val = kwargs.get("threshold", 0.5)
    threshold = float(coerce_float(threshold_val)) if threshold_val is not None else 0.5
    value_range_val = kwargs.get("value_range", (0, 255))
    value_range = value_range_val if isinstance(value_range_val, tuple) else (0, 255)
    return _ensure_arr(
        arr_mod.where(
            arr_mod.greater_equal(img_arr, threshold),
            arr_mod.add(arr_mod.subtract(value_range[1], img_arr), value_range[0]),
            img_arr,
        )
    )


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
