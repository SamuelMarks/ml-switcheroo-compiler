# ruff: noqa: E501
"""Signal utilities."""

import typing

from ml_switcheroo_compiler.ops.configs import BlurConfig


def _generate_gaussian_kernel(np_mod: object, kernel_size: tuple[int, int], sigma: tuple[float, float]) -> object:
    """Evaluate _generate_gaussian_kernel operation.

    Args:
        np_mod (object): The np_mod parameter.
        kernel_size (object): The kernel_size parameter.
        sigma (object): The sigma parameter.

    Returns:
        object: Result.
    """
    kx, ky = kernel_size
    sx, sy = sigma
    ax = np_mod.arange(-kx // 2 + 1.0, kx // 2 + 1.0)
    ay = np_mod.arange(-ky // 2 + 1.0, ky // 2 + 1.0)
    xx, yy = np_mod.meshgrid(ax, ay)
    kernel = np_mod.exp(-(xx**2 / (2.0 * sx**2) + yy**2 / (2.0 * sy**2)))
    return kernel / np_mod.sum(kernel)


def _apply_conv2d_batch(np_mod: object, imgs: object, kernel: object, mode: str) -> object:
    """Evaluate _apply_conv2d_batch operation.

    Args:
        np_mod (object): The np_mod parameter.
        imgs (object): The imgs parameter.
        kernel (object): The kernel parameter.
        mode (str): The mode parameter.

    Returns:
        object: Result.

    Raises:
        ValueError: An exception.
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
    """Evaluate _get_blur_config operation.

    Args:
        kwargs (dict): The kwargs parameter.
        config_obj (object): The config_obj parameter.

    Returns:
        object: Result.
    """
    if config_obj is None:
        return BlurConfig(
            kernel_size=kwargs.get("kernel_size", (3, 3)),
            sigma=kwargs.get("sigma", (1.0, 1.0)),
            data_format=kwargs.get("data_format", None),
        )
    return config_obj


def gaussian_blur_eager(backend_module: object, images: object, config_obj: typing.Optional[object] = None, **kwargs: object) -> object:
    """Evaluate gaussian_blur_eager operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        config_obj (object): The config_obj parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    config = _get_blur_config(kwargs, config_obj)
    kernel = _generate_gaussian_kernel(backend_module, config.kernel_size, config.sigma)
    return _apply_conv2d_batch(backend_module, images, kernel, "reflect")


def _apply_median_filter_batch(np_mod: object, imgs: object, kernel_size: tuple[int, int], padding: str) -> object:
    """Evaluate _apply_median_filter_batch operation.

    Args:
        np_mod (object): The np_mod parameter.
        imgs (object): The imgs parameter.
        kernel_size (object): The kernel_size parameter.
        padding (str): The padding parameter.

    Returns:
        object: Result.

    Raises:
        ValueError: An exception.
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
    """Evaluate median_filter_eager operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        kernel_size (object): The kernel_size parameter.
        padding (str): The padding parameter.
        data_format (object): The data_format parameter.

    Returns:
        object: Result.
    """
    return _apply_median_filter_batch(backend_module, images, kernel_size, padding)
