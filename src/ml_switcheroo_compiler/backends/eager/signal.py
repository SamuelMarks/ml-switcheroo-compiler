"""signal.py module."""

from typing import Callable, Optional, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.ops.configs import BlurConfig


def _generate_gaussian_kernel(np_mod: EagerValue, kernel_size: tuple[int, int], sigma: tuple[float, float]) -> EagerValue:
    """Execute the eager generate gaussian kernel operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        kernel_size (tuple[int, int]): The kernel_size parameter.
        sigma (tuple[float, float]): The sigma parameter.

    Returns:
            EagerValue: Result.
    """
    kx, ky = kernel_size
    sx, sy = sigma
    ax = getattr(np_mod, "arange")(-kx // 2 + 1.0, kx // 2 + 1.0)
    ay = getattr(np_mod, "arange")(-ky // 2 + 1.0, ky // 2 + 1.0)
    xx, yy = getattr(np_mod, "meshgrid")(ax, ay)
    kernel = getattr(np_mod, "exp")(-(xx**2 / (2.0 * sx**2) + yy**2 / (2.0 * sy**2)))
    return cast(EagerValue, kernel / getattr(np_mod, "sum")(kernel))


def _apply_conv2d_batch(np_mod: EagerValue, imgs: EagerValue, kernel: EagerValue, mode: str) -> EagerValue:
    """Execute the eager apply conv2d batch operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        imgs: The imgs parameter.
        kernel: The kernel parameter.
        mode (str): The mode parameter.

    Returns:
            EagerValue: Result.

    Raises:
        ValueError: An exception.
    """
    # Basic implementation for 4D images (B, C, H, W)
    if getattr(imgs, "ndim") != 4:
        raise ValueError("Images must be 4D")
    B, C, H, W = cast(tuple[int, ...], getattr(imgs, "shape"))
    kH, kW = cast(tuple[int, ...], getattr(kernel, "shape"))
    pad_h = kH // 2
    pad_w = kW // 2
    padded = getattr(np_mod, "pad")(imgs, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = getattr(np_mod, "zeros_like")(imgs)
    # Simple nested loops or vectorization
    # Since np_mod might be cupy or jax, use standard indexing
    for y in range(H):
        for x in range(W):
            region = padded[:, :, y : y + kH, x : x + kW]
            # kernel shape is (kH, kW), expand to (1, 1, kH, kW)
            expanded_kernel = getattr(np_mod, "expand_dims")(getattr(np_mod, "expand_dims")(kernel, 0), 0)
            out[:, :, y, x] = getattr(np_mod, "sum")(region * expanded_kernel, axis=(-2, -1))
    return cast(EagerValue, out)


def _get_blur_config(kwargs: dict[str, EagerValue], config_obj: Optional[BlurConfig]) -> BlurConfig:
    """Execute the eager get blur config operation on the active backend array.

    Args:
        kwargs (dict[str, object]): The kwargs parameter.
        config_obj: The config_obj parameter.

    Returns:
            BlurConfig: Result.
    """
    if config_obj is None:
        return cast(
            BlurConfig,
            cast("Callable[..., EagerTensorProtocol]", BlurConfig)(
                kernel_size=kwargs.get("kernel_size", (3, 3)) if hasattr(kwargs, "get") else (3, 3),
                sigma=kwargs.get("sigma", (1.0, 1.0)) if hasattr(kwargs, "get") else (1.0, 1.0),
                data_format=str(kwargs.get("data_format", None)) if hasattr(kwargs, "get") else None,
            ),
        )
    return config_obj


def gaussian_blur_eager(backend_module: EagerValue, images: EagerValue, config_obj: Optional[BlurConfig] = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager gaussian blur eager operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        images: The images parameter.
        config_obj: The config_obj parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    config = _get_blur_config(kwargs, config_obj)
    kernel = cast("Callable[..., EagerTensorProtocol]", _generate_gaussian_kernel)(backend_module, config.kernel_size, config.sigma)
    return _apply_conv2d_batch(backend_module, images, kernel, "reflect")


def _apply_median_filter_batch(np_mod: EagerValue, imgs: EagerValue, kernel_size: tuple[int, int], padding: str) -> EagerValue:
    """Execute the eager apply median filter batch operation on the active backend array.

    Args:
        np_mod: The np_mod parameter.
        imgs: The imgs parameter.
        kernel_size (tuple[int, int]): The kernel_size parameter.
        padding (str): The padding parameter.

    Returns:
            EagerValue: Result.

    Raises:
        ValueError: An exception.
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
            # flattening the last two dims and taking median
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
    """Execute the eager median filter eager operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        images: The images parameter.
        kernel_size (tuple[int, int]): The kernel_size parameter.
        padding (str): The padding parameter.
        data_format: The data_format parameter.

    Returns:
            EagerValue: Result.
    """
    return _apply_median_filter_batch(backend_module, images, kernel_size, padding)
