"""signal.py module."""

from typing import Any, Callable, Optional

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Signal utilities."""


from ml_switcheroo_compiler.ops.configs import BlurConfig


def _generate_gaussian_kernel(np_mod: Any, kernel_size: tuple[int, int], sigma: tuple[float, float]) -> Any:
    """Evaluate _generate_gaussian_kernel operation.

    Args:
        np_mod (object): The np_mod parameter.
        kernel_size (object): The kernel_size parameter.
        sigma (object): The sigma parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    kx, ky = kernel_size
    sx, sy = sigma
    ax: Any = np_mod.arange(-kx // 2 + 1.0, kx // 2 + 1.0)
    ay: Any = np_mod.arange(-ky // 2 + 1.0, ky // 2 + 1.0)
    xx, yy = np_mod.meshgrid(ax, ay)
    kernel: Any = np_mod.exp(-(xx**2 / (2.0 * sx**2) + yy**2 / (2.0 * sy**2)))
    return kernel / np_mod.sum(kernel)


def _apply_conv2d_batch(np_mod: Any, imgs: Any, kernel: Any, mode: str) -> Any:
    """Evaluate _apply_conv2d_batch operation.

    Args:
        np_mod (object): The np_mod parameter.
        imgs (object): The imgs parameter.
        kernel (object): The kernel parameter.
        mode (str): The mode parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    # Basic implementation for 4D images (B, C, H, W)
    if imgs.ndim != 4:
        raise ValueError("Images must be 4D")
    B, C, H, W = imgs.shape
    kH, kW = kernel.shape
    pad_h: Any = kH // 2
    pad_w: Any = kW // 2
    padded: Any = np_mod.pad(imgs, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out: Any = np_mod.zeros_like(imgs)
    # Simple nested loops or vectorization
    # Since np_mod might be cupy or jax, use standard indexing
    for y in range(H):
        for x in range(W):
            region: Any = padded[:, :, y : y + kH, x : x + kW]
            # kernel shape is (kH, kW), expand to (1, 1, kH, kW)
            expanded_kernel: Any = np_mod.expand_dims(np_mod.expand_dims(kernel, 0), 0)
            out[:, :, y, x] = np_mod.sum(region * expanded_kernel, axis=(-2, -1))
    return out


def _get_blur_config(kwargs: dict[str, object], config_obj: Optional[object]) -> Any:
    """Evaluate _get_blur_config operation.

    Args:
        kwargs (dict): The kwargs parameter.
        config_obj (object): The config_obj parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if config_obj is None:
        return BlurConfig(
            kernel_size=kwargs.get("kernel_size", (3, 3)),
            sigma=kwargs.get("sigma", (1.0, 1.0)),
            data_format=kwargs.get("data_format", None),
        )
    return config_obj


def gaussian_blur_eager(backend_module: Any, images: Any, config_obj: Optional[object] = None, **kwargs: Any) -> Any:
    """Evaluate gaussian_blur_eager operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        config_obj (object): The config_obj parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config: Any = _get_blur_config(kwargs, config_obj)
    kernel: Any = _generate_gaussian_kernel(backend_module, config.kernel_size, config.sigma)
    return _apply_conv2d_batch(backend_module, images, kernel, "reflect")


def _apply_median_filter_batch(np_mod: Any, imgs: Any, kernel_size: tuple[int, int], padding: str) -> Any:
    """Evaluate _apply_median_filter_batch operation.

    Args:
        np_mod (object): The np_mod parameter.
        imgs (object): The imgs parameter.
        kernel_size (object): The kernel_size parameter.
        padding (str): The padding parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    if imgs.ndim != 4:
        raise ValueError("Images must be 4D")
    B, C, H, W = imgs.shape
    kH, kW = kernel_size
    pad_h: Any = kH // 2
    pad_w: Any = kW // 2
    padded: Any = np_mod.pad(imgs, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant" if padding == "same" else "reflect")
    out: Any = np_mod.zeros_like(imgs)
    for y in range(H):
        for x in range(W):
            region: Any = padded[:, :, y : y + kH, x : x + kW]
            # flattening the last two dims and taking median
            flat_region: Any = np_mod.reshape(region, (B, C, kH * kW))
            out[:, :, y, x] = np_mod.median(flat_region, axis=-1)
    return out


def median_filter_eager(
    backend_module: Any,
    images: Any,
    kernel_size: tuple[int, int],
    padding: str = "same",
    data_format: Any = None,
) -> Any:
    """Evaluate median_filter_eager operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        kernel_size (object): The kernel_size parameter.
        padding (str): The padding parameter.
        data_format (object): The data_format parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return _apply_median_filter_batch(backend_module, images, kernel_size, padding)
