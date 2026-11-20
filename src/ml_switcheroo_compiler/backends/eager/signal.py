"""Eager signal and 2D spatial filtering operations for backend array modules."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float_list,
    coerce_shape,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.core.shape import OpValue as OpValue
from ml_switcheroo_compiler.ops.configs import BlurConfig


def _generate_gaussian_kernel(
    np_mod: BackendArg, kernel_size: tuple[int, int], sigma: tuple[float, float]
) -> BackendArg:
    """Generate a normalized 2D Gaussian filter kernel on the active backend module.

    Args:
        np_mod (BackendArg): Active backend array module.
        kernel_size (tuple[int, int]): Spatial height and width of the Gaussian kernel.
        sigma (tuple[float, float]): Standard deviation along x and y dimensions.

    Returns:
        BackendArg: Normalized 2D Gaussian kernel array.
    """
    kx, ky = kernel_size
    sx, sy = sigma
    ax = adapt_backend_module(np_mod).require_symbol("arange")(-kx // 2 + 1.0, kx // 2 + 1.0)
    ay = adapt_backend_module(np_mod).require_symbol("arange")(-ky // 2 + 1.0, ky // 2 + 1.0)
    xx, yy = adapt_backend_module(np_mod).require_symbol("meshgrid")(ax, ay)
    kernel = adapt_backend_module(np_mod).require_symbol("exp")(-(xx**2 / (2.0 * sx**2) + yy**2 / (2.0 * sy**2)))
    return kernel / adapt_backend_module(np_mod).require_symbol("sum")(kernel)


def _apply_conv2d_batch(np_mod: BackendArg, imgs: BackendArg, kernel: BackendArg, mode: str) -> BackendArg:
    """Apply a 2D spatial filter kernel across a 4D NCHW batch of images.

    Args:
        np_mod (BackendArg): Active backend array module.
        imgs (BackendArg): Input 4D image batch tensor of shape (B, C, H, W).
        kernel (BackendArg): 2D spatial convolution kernel of shape (kH, kW).
        mode (str): Boundary padding mode (e.g. 'reflect').

    Returns:
        BackendArg: Filtered 4D image batch tensor.

    Raises:
        ValueError: If the input images tensor is not 4-dimensional.
    """
    del mode
    if adapt_backend_module(imgs).require_symbol("ndim") != 4:
        raise ValueError("Images must be 4D")
    _b, _c, H, W = coerce_shape(adapt_backend_module(imgs).require_symbol("shape"))
    kH, kW = coerce_shape(adapt_backend_module(kernel).require_symbol("shape"))
    pad_h = kH // 2
    pad_w = kW // 2
    padded = adapt_backend_module(np_mod).require_symbol("pad")(
        imgs, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="reflect"
    )
    out = adapt_backend_module(np_mod).require_symbol("zeros_like")(imgs)
    for y in range(H):
        for x in range(W):
            region = padded[:, :, y : y + kH, x : x + kW]
            expanded_kernel = adapt_backend_module(np_mod).require_symbol("expand_dims")(
                adapt_backend_module(np_mod).require_symbol("expand_dims")(kernel, 0), 0
            )
            out[:, :, y, x] = adapt_backend_module(np_mod).require_symbol("sum")(
                region * expanded_kernel, axis=(-2, -1)
            )
    return out


def _get_blur_config(kwargs: dict[str, BackendArg], config_obj: BlurConfig | None) -> BlurConfig:
    """Resolve or construct a BlurConfig instance from keyword arguments.

    Args:
        kwargs (dict[str, BackendArg]): Keyword arguments dictionary.
        config_obj (Optional[BlurConfig]): Explicit BlurConfig instance if provided.

    Returns:
        BlurConfig: Resolved Gaussian blur configuration.
    """
    if config_obj is None:
        ks = kwargs.get("kernel_size", (3, 3)) if adapt_backend_module(kwargs).has_symbol("get") else (3, 3)
        sg = kwargs.get("sigma", (1.0, 1.0)) if adapt_backend_module(kwargs).has_symbol("get") else (1.0, 1.0)
        df = str(kwargs.get("data_format", None)) if adapt_backend_module(kwargs).has_symbol("get") else None
        ks_seq = coerce_shape(ks)
        ks_pair: tuple[int, int] = (ks_seq[0], ks_seq[1]) if len(ks_seq) >= 2 else (3, 3)
        sg_seq = coerce_float_list(sg)
        sg_pair: tuple[float, float] = (sg_seq[0], sg_seq[1]) if len(sg_seq) >= 2 else (1.0, 1.0)
        return BlurConfig(
            kernel_size=ks_pair,
            sigma=sg_pair,
            data_format=df,
        )
    return config_obj


def gaussian_blur_eager(
    backend_module: BackendArg, images: BackendArg, config_obj: BlurConfig | None = None, **kwargs: BackendArg
) -> BackendArg:
    """Apply eager 2D Gaussian blur filtering to a 4D image batch.

    Args:
        backend_module (BackendArg): Active backend array module.
        images (BackendArg): Input 4D image batch tensor.
        config_obj (Optional[BlurConfig]): Optional blur configuration object.
        **kwargs (BackendArg): Additional blur configuration keyword parameters.

    Returns:
        BackendArg: Blurred 4D image batch tensor.
    """
    config = _get_blur_config(kwargs, config_obj)
    ks_seq = coerce_shape(config.kernel_size)
    ks_pair: tuple[int, int] = (ks_seq[0], ks_seq[1]) if len(ks_seq) >= 2 else (3, 3)
    sg_seq = coerce_float_list(config.sigma)
    sg_pair: tuple[float, float] = (sg_seq[0], sg_seq[1]) if len(sg_seq) >= 2 else (1.0, 1.0)
    kernel = _generate_gaussian_kernel(backend_module, ks_pair, sg_pair)
    return _apply_conv2d_batch(backend_module, images, kernel, "reflect")


def _apply_median_filter_batch(
    np_mod: BackendArg, imgs: BackendArg, kernel_size: tuple[int, int], padding: str
) -> BackendArg:
    """Apply a 2D spatial median filter across a 4D NCHW batch of images.

    Args:
        np_mod (BackendArg): Active backend array module.
        imgs (BackendArg): Input 4D image batch tensor of shape (B, C, H, W).
        kernel_size (tuple[int, int]): Spatial window size (kH, kW).
        padding (str): Padding mode ('same' or 'reflect').

    Returns:
        BackendArg: Median-filtered 4D image batch tensor.

    Raises:
        ValueError: If the input images tensor is not 4-dimensional.
    """
    if adapt_backend_module(imgs).require_symbol("ndim") != 4:
        raise ValueError("Images must be 4D")
    B, C, H, W = coerce_shape(adapt_backend_module(imgs).require_symbol("shape"))
    kH, kW = kernel_size
    pad_h = kH // 2
    pad_w = kW // 2
    padded = adapt_backend_module(np_mod).require_symbol("pad")(
        imgs, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant" if padding == "same" else "reflect"
    )
    out = adapt_backend_module(np_mod).require_symbol("zeros_like")(imgs)
    for y in range(H):
        for x in range(W):
            region = padded[:, :, y : y + kH, x : x + kW]
            flat_region = adapt_backend_module(np_mod).require_symbol("reshape")(region, (B, C, kH * kW))
            out[:, :, y, x] = adapt_backend_module(np_mod).require_symbol("median")(flat_region, axis=-1)
    return out


def median_filter_eager(
    backend_module: BackendArg,
    images: BackendArg,
    kernel_size: tuple[int, int],
    padding: str = "same",
    data_format: BackendArg | None = None,
) -> BackendArg:
    """Apply eager 2D median filtering to a 4D image batch.

    Args:
        backend_module (BackendArg): Active backend array module.
        images (BackendArg): Input 4D image batch tensor.
        kernel_size (tuple[int, int]): Spatial window size (kH, kW).
        padding (str): Boundary padding mode ('same' or 'reflect').
        data_format (Optional[BackendArg]): Optional channel layout format.

    Returns:
        BackendArg: Median-filtered 4D image batch tensor.
    """
    del data_format
    return _apply_median_filter_batch(backend_module, images, kernel_size, padding)
