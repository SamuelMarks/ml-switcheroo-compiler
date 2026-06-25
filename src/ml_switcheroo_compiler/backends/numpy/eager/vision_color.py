# ruff: noqa: F405, F403
"""Shared vision utilities and ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
import numpy as np


@numpy_eager_registry.register("AdjustBrightness")
def _np_adjust_brightness(
    backend_module: object, images: object, delta: float, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        delta: Arg.
        kwargs: Arg.
    """
    return np.clip(images + delta, 0.0, 1.0)


@numpy_eager_registry.register("AdjustContrast")
def _np_adjust_contrast(
    backend_module: object, images: object, contrast_factor: float, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        contrast_factor: Arg.
        kwargs: Arg.
    """
    mean = np.mean(images, axis=(-3, -2), keepdims=True)
    return np.clip((images - mean) * contrast_factor + mean, 0.0, 1.0)


@numpy_eager_registry.register("AdjustHue")
def _np_adjust_hue(
    backend_module: object, images: object, delta: float, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        delta: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("AdjustSaturation")
def _np_adjust_saturation(
    backend_module: object, images: object, saturation_factor: float, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        saturation_factor: Arg.
        kwargs: Arg.
    """
    gray = _np_rgb_to_grayscale(backend_module, images)
    return np.clip(gray + (images - gray) * saturation_factor, 0.0, 1.0)


@numpy_eager_registry.register("AutoContrast")
def _np_auto_contrast(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    value_range = kwargs.get("value_range", (0, 255))
    low = np.min(images, axis=(-3, -2), keepdims=True)
    high = np.max(images, axis=(-3, -2), keepdims=True)
    diff = high - low
    diff = np.where(diff == 0.0, 1.0, diff)
    out = (images - low) / diff
    return np.clip(
        out * (value_range[1] - value_range[0]) + value_range[0], value_range[0], value_range[1]
    ).astype(images.dtype)


@numpy_eager_registry.register("Equalization")
def _np_equalization(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    images_uint8 = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    out = np.empty_like(images_uint8)
    for b in range(images.shape[0]):
        for c in range(images.shape[-1]):
            hist, _ = np.histogram(images_uint8[b, ..., c].flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            if cdf_m.max() - cdf_m.min() == 0:  # pragma: no branch
                out[b, ..., c] = images_uint8[b, ..., c]  # pragma: no cover
            else:
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf = np.ma.filled(cdf_m, 0).astype("uint8")
                out[b, ..., c] = cdf[images_uint8[b, ..., c]]
    return out.astype(images.dtype) / 255.0


@numpy_eager_registry.register("Invert")
def _np_invert(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    value_range = kwargs.get("value_range", (0, 255))
    return value_range[1] - images + value_range[0]


@numpy_eager_registry.register("Posterize")
def _np_posterize(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    bits = kwargs.get("bits", 4)
    shift = 8 - bits
    images_uint8 = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    posterized = np.bitwise_and(images_uint8, np.array(~((1 << shift) - 1), dtype=np.uint8))
    return posterized.astype(images.dtype) / 255.0


@numpy_eager_registry.register("RgbToGrayscale")
def _np_rgb_to_grayscale(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
    """
    np_mod = __import__("numpy")
    data_format = kwargs.get("data_format", "channels_last")
    from ml_switcheroo_compiler.backends.eager.utils import _to_channels_last, _from_channels_last

    imgs = _to_channels_last(np_mod, images, data_format)
    # rgb to grayscale weights
    weights = np_mod.array([0.2989, 0.5870, 0.1140], dtype=imgs.dtype)
    gray = np_mod.sum(imgs * weights, axis=-1, keepdims=True)
    gray = _from_channels_last(np_mod, gray, data_format)
    return gray


@numpy_eager_registry.register("Solarize")
def _np_solarize(backend_module: object, images: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        kwargs: Arg.
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
