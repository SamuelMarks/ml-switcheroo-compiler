"""Eager signal and 2D spatial filtering operations for backend array modules."""

from typing import Optional, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.ops.configs import BlurConfig


def _generate_gaussian_kernel(np_mod: EagerValue, kernel_size: tuple[int, int], sigma: tuple[float, float]) -> EagerValue:
    """Generate a normalized 2D Gaussian filter kernel on the active backend module.

    Args:
        np_mod (EagerValue): Active backend array module.
        kernel_size (tuple[int, int]): Spatial height and width of the Gaussian kernel.
        sigma (tuple[float, float]): Standard deviation along x and y dimensions.

    Returns:
        EagerValue: Normalized 2D Gaussian kernel array.
    """
    kx, ky = kernel_size
    sx, sy = sigma
    ax = getattr(np_mod, "arange")(-kx // 2 + 1.0, kx // 2 + 1.0)
    ay = getattr(np_mod, "arange")(-ky // 2 + 1.0, ky // 2 + 1.0)
    xx, yy = getattr(np_mod, "meshgrid")(ax, ay)
    kernel = getattr(np_mod, "exp")(-(xx**2 / (2.0 * sx**2) + yy**2 / (2.0 * sy**2)))
    return cast(EagerValue, kernel / getattr(np_mod, "sum")(kernel))


def _apply_conv2d_batch(np_mod: EagerValue, imgs: EagerValue, kernel: EagerValue, mode: str) -> EagerValue:
    """Apply a 2D spatial filter kernel across a 4D NCHW batch of images.

    Args:
        np_mod (EagerValue): Active backend array module.
        imgs (EagerValue): Input 4D image batch tensor of shape (B, C, H, W).
        kernel (EagerValue): 2D spatial convolution kernel of shape (kH, kW).
        mode (str): Boundary padding mode (e.g. 'reflect').

    Returns:
        EagerValue: Filtered 4D image batch tensor.

    Raises:
        ValueError: If the input images tensor is not 4-dimensional.
    """
    del mode
    if getattr(imgs, "ndim") != 4:
        raise ValueError("Images must be 4D")
    _b, _c, H, W = cast(tuple[int, ...], getattr(imgs, "shape"))
    kH, kW = cast(tuple[int, ...], getattr(kernel, "shape"))
    pad_h = kH // 2
    pad_w = kW // 2
    padded = getattr(np_mod, "pad")(imgs, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = getattr(np_mod, "zeros_like")(imgs)
    for y in range(H):
        for x in range(W):
            region = padded[:, :, y : y + kH, x : x + kW]
            expanded_kernel = getattr(np_mod, "expand_dims")(getattr(np_mod, "expand_dims")(kernel, 0), 0)
            out[:, :, y, x] = getattr(np_mod, "sum")(region * expanded_kernel, axis=(-2, -1))
    return cast(EagerValue, out)


def _get_blur_config(kwargs: dict[str, EagerValue], config_obj: Optional[BlurConfig]) -> BlurConfig:
    """Resolve or construct a BlurConfig instance from keyword arguments.

    Args:
        kwargs (dict[str, EagerValue]): Keyword arguments dictionary.
        config_obj (Optional[BlurConfig]): Explicit BlurConfig instance if provided.

    Returns:
        BlurConfig: Resolved Gaussian blur configuration.
    """
    if config_obj is None:
        ks = kwargs.get("kernel_size", (3, 3)) if hasattr(kwargs, "get") else (3, 3)
        sg = kwargs.get("sigma", (1.0, 1.0)) if hasattr(kwargs, "get") else (1.0, 1.0)
        df = str(kwargs.get("data_format", None)) if hasattr(kwargs, "get") else None
        return BlurConfig(
            kernel_size=cast("tuple[int, int]", ks),
            sigma=cast("tuple[float, float]", sg),
            data_format=df,
        )
    return config_obj


def gaussian_blur_eager(backend_module: EagerValue, images: EagerValue, config_obj: Optional[BlurConfig] = None, **kwargs: EagerValue) -> EagerValue:
    """Apply eager 2D Gaussian blur filtering to a 4D image batch.

    Args:
        backend_module (EagerValue): Active backend array module.
        images (EagerValue): Input 4D image batch tensor.
        config_obj (Optional[BlurConfig]): Optional blur configuration object.
        **kwargs (EagerValue): Additional blur configuration keyword parameters.

    Returns:
        EagerValue: Blurred 4D image batch tensor.
    """
    config = _get_blur_config(kwargs, config_obj)
    kernel = _generate_gaussian_kernel(backend_module, cast("tuple[int, int]", config.kernel_size), cast("tuple[float, float]", config.sigma))
    return _apply_conv2d_batch(backend_module, images, kernel, "reflect")


def _apply_median_filter_batch(np_mod: EagerValue, imgs: EagerValue, kernel_size: tuple[int, int], padding: str) -> EagerValue:
    """Apply a 2D spatial median filter across a 4D NCHW batch of images.

    Args:
        np_mod (EagerValue): Active backend array module.
        imgs (EagerValue): Input 4D image batch tensor of shape (B, C, H, W).
        kernel_size (tuple[int, int]): Spatial window size (kH, kW).
        padding (str): Padding mode ('same' or 'reflect').

    Returns:
        EagerValue: Median-filtered 4D image batch tensor.

    Raises:
        ValueError: If the input images tensor is not 4-dimensional.
    """
    if getattr(imgs, "ndim") != 4:
        raise ValueError("Images must be 4D")
    B, C, H, W = cast(tuple[int, ...], getattr(imgs, "shape"))
    kH, kW = kernel_size
    pad_h = kH // 2
    pad_w = kW // 2
    padded = getattr(np_mod, "pad")(imgs, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant" if padding == "same" else "reflect")
    out = getattr(np_mod, "zeros_like")(imgs)
    for y in range(H):
        for x in range(W):
            region = padded[:, :, y : y + kH, x : x + kW]
            flat_region = getattr(np_mod, "reshape")(region, (B, C, kH * kW))
            out[:, :, y, x] = getattr(np_mod, "median")(flat_region, axis=-1)
    return cast(EagerValue, out)


def median_filter_eager(
    backend_module: EagerValue,
    images: EagerValue,
    kernel_size: tuple[int, int],
    padding: str = "same",
    data_format: Optional[EagerValue] = None,
) -> EagerValue:
    """Apply eager 2D median filtering to a 4D image batch.

    Args:
        backend_module (EagerValue): Active backend array module.
        images (EagerValue): Input 4D image batch tensor.
        kernel_size (tuple[int, int]): Spatial window size (kH, kW).
        padding (str): Boundary padding mode ('same' or 'reflect').
        data_format (Optional[EagerValue]): Optional channel layout format.

    Returns:
        EagerValue: Median-filtered 4D image batch tensor.
    """
    del data_format
    return _apply_median_filter_batch(backend_module, images, kernel_size, padding)
