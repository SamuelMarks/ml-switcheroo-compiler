# ruff: noqa: E501
"""Signal utilities."""

import typing

from ml_switcheroo_compiler.ops.configs import BlurConfig


def _generate_gaussian_kernel(np_mod: object, kernel_size: tuple[int, int], sigma: tuple[float, float]) -> object:
    """Evaluate and process the generate gaussian kernel operation.

    Args:
        np_mod (object): Required parameter for np_mod.
        kernel_size (tuple): Required parameter for kernel_size.
        sigma (tuple): Required parameter for sigma.

    Returns:
        object: The evaluated or processed output.
    """
    kx, ky = kernel_size
    sx, sy = sigma
    ax = np_mod.arange(-kx // 2 + 1.0, kx // 2 + 1.0)
    ay = np_mod.arange(-ky // 2 + 1.0, ky // 2 + 1.0)
    xx, yy = np_mod.meshgrid(ax, ay)
    kernel = np_mod.exp(-(xx**2 / (2.0 * sx**2) + yy**2 / (2.0 * sy**2)))
    return kernel / np_mod.sum(kernel)


def _apply_conv2d_batch(np_mod: object, imgs: object, kernel: object, mode: str) -> object:
    """Evaluate and process the apply conv2d batch operation.

    Args:
        np_mod (object): Required parameter for np_mod.
        imgs (object): Required parameter for imgs.
        kernel (object): Required parameter for kernel.
        mode (str): Required parameter for mode.

    Returns:
        object: The evaluated or processed output.
    """
    # Basic implementation for 4D images (B, C, H, W)
    if imgs.ndim != 4:
        raise ValueError("Images must be 4D")
    B, C, H, W = imgs.shape
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2
    padded = np_mod.pad(imgs, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np_mod.zeros_like(imgs)
    # Simple nested loops or vectorization
    # Since np_mod might be cupy or jax, use standard indexing
    for y in range(H):
        for x in range(W):
            region = padded[:, :, y : y + kH, x : x + kW]
            # kernel shape is (kH, kW), expand to (1, 1, kH, kW)
            expanded_kernel = np_mod.expand_dims(np_mod.expand_dims(kernel, 0), 0)
            out[:, :, y, x] = np_mod.sum(region * expanded_kernel, axis=(-2, -1))
    return out


def _get_blur_config(kwargs: dict, config_obj: typing.Optional[object]) -> object:
    """Retrieve the blur config property or mapping.

    Args:
        kwargs (dict): Required parameter for kwargs.
        config_obj (Any): Required parameter for config_obj.

    Returns:
        object: The evaluated or processed output.
    """
    if config_obj is None:
        return BlurConfig(
            kernel_size=kwargs.get("kernel_size", (3, 3)),
            sigma=kwargs.get("sigma", (1.0, 1.0)),
            data_format=kwargs.get("data_format", None),
        )
    return config_obj


def gaussian_blur_eager(backend_module: object, images: object, config_obj: typing.Optional[object] = None, **kwargs: object) -> object:
    """Evaluate gaussian blur eagerly."""
    config = _get_blur_config(kwargs, config_obj)
    kernel = _generate_gaussian_kernel(backend_module, config.kernel_size, config.sigma)
    return _apply_conv2d_batch(backend_module, images, kernel, "reflect")


def _apply_median_filter_batch(np_mod: object, imgs: object, kernel_size: tuple[int, int], padding: str) -> object:
    """Evaluate and process the apply median filter batch operation.

    Args:
        np_mod (object): Required parameter for np_mod.
        imgs (object): Required parameter for imgs.
        kernel_size (tuple): Required parameter for kernel_size.
        padding (str): Required parameter for padding.

    Returns:
        object: The evaluated or processed output.
    """
    if imgs.ndim != 4:
        raise ValueError("Images must be 4D")
    B, C, H, W = imgs.shape
    kH, kW = kernel_size
    pad_h = kH // 2
    pad_w = kW // 2
    padded = np_mod.pad(imgs, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant" if padding == "same" else "reflect")
    out = np_mod.zeros_like(imgs)
    for y in range(H):
        for x in range(W):
            region = padded[:, :, y : y + kH, x : x + kW]
            # flattening the last two dims and taking median
            flat_region = np_mod.reshape(region, (B, C, kH * kW))
            out[:, :, y, x] = np_mod.median(flat_region, axis=-1)
    return out


def median_filter_eager(
    backend_module: object,
    images: object,
    kernel_size: tuple[int, int],
    padding: str = "same",
    data_format: object = None,
) -> object:
    """Evaluate median filter eagerly."""
    return _apply_median_filter_batch(backend_module, images, kernel_size, padding)
