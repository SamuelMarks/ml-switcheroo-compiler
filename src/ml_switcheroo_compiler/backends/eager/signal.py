"""Signal utilities."""

import typing
from dataclasses import dataclass

from ml_switcheroo_compiler.backends.eager.utils import (
    _from_channels_last,
    _from_numpy_array,
    _to_channels_last,
    _to_numpy_array,
)


def _generate_gaussian_kernel(
    np_mod: object, kernel_size: tuple[int, int], sigma: tuple[float, float]
) -> object:
    ky, kx = kernel_size
    sy, sx = sigma
    y = np_mod.arange(-ky // 2 + 1, ky // 2 + 1)
    x = np_mod.arange(-kx // 2 + 1, kx // 2 + 1)
    yy, xx = np_mod.meshgrid(y, x, indexing="ij")
    kernel = np_mod.exp(-(yy**2 / (2.0 * sy**2) + xx**2 / (2.0 * sx**2)))
    return kernel / np_mod.sum(kernel)


def _apply_conv2d_batch(np_mod: object, imgs: object, kernel: object, mode: str) -> object:
    import scipy.signal

    B, H, W, C = imgs.shape
    ky, kx = kernel.shape

    if mode == "valid":
        out = np_mod.zeros((B, H - ky + 1, W - kx + 1, C), dtype=imgs.dtype)
    else:
        out = np_mod.zeros_like(imgs)

    for b in range(B):
        for c in range(C):
            out[b, ..., c] = scipy.signal.convolve2d(
                imgs[b, ..., c], kernel, mode=mode, boundary="fill", fillvalue=0.0
            )
    return out


def _get_blur_config(kwargs: dict, config_obj: typing.Optional[object]) -> object:
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import BlurConfig

        return BlurConfig(
            kernel_size=kwargs.get("kernel_size", (3, 3)),
            sigma=kwargs.get("sigma", (1.0, 1.0)),
            data_format=kwargs.get("data_format", None),
        )
    return config_obj


def gaussian_blur_eager(
    backend_module: object,
    images: object,
    config_obj: typing.Optional[object] = None,
    **kwargs: object,
) -> object:
    """Evaluate gaussian blur eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    config_obj = _get_blur_config(kwargs, config_obj)
    padding = kwargs.get("padding", "same")

    imgs = _to_numpy_array(np_mod, images, name)
    kernel = _generate_gaussian_kernel(np_mod, config_obj.kernel_size, config_obj.sigma)

    original_ndim = imgs.ndim
    if original_ndim == 3:
        imgs = imgs[None, ...]

    imgs = _to_channels_last(np_mod, imgs, config_obj.data_format)

    mode = "same" if padding == "same" else "valid"
    out = _apply_conv2d_batch(np_mod, imgs, kernel, mode)

    out = _from_channels_last(np_mod, out, config_obj.data_format)

    if original_ndim == 3:
        out = out[0]

    return _from_numpy_array(backend_module, out, name, images)


@dataclass
class FilterConfig:
    """Configuration for filtering."""

    ky: int
    kx: int
    padding: str


def _apply_median_filter_channel(imgs: object, out: object, config: FilterConfig, b: int) -> None:
    import scipy.ndimage

    C = imgs.shape[-1]
    for c in range(C):
        filtered = scipy.ndimage.median_filter(
            imgs[b, ..., c], size=(config.ky, config.kx), mode="constant", cval=0.0
        )
        if config.padding == "valid":
            pad_y_top = config.ky // 2
            pad_x_left = config.kx // 2
            filtered = filtered[
                pad_y_top : pad_y_top + out.shape[1], pad_x_left : pad_x_left + out.shape[2]
            ]
        out[b, ..., c] = filtered


def _apply_median_filter_batch(
    np_mod: object, imgs: object, kernel_size: tuple[int, int], padding: str
) -> object:

    B, H, W, C = imgs.shape
    ky, kx = kernel_size

    if padding == "valid":
        out = np_mod.zeros((B, H - ky + 1, W - kx + 1, C), dtype=imgs.dtype)
    else:
        out = np_mod.zeros_like(imgs)

    config = FilterConfig(ky=ky, kx=kx, padding=padding)
    for b in range(B):
        _apply_median_filter_channel(imgs, out, config, b)
    return out


def median_filter_eager(
    backend_module: object,
    images: object,
    kernel_size: tuple[int, int],
    padding: str = "same",
    data_format: object = None,
) -> object:
    """Evaluate median filter eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    imgs = _to_numpy_array(np_mod, images, name)

    original_ndim = imgs.ndim
    if original_ndim == 3:
        imgs = imgs[None, ...]

    imgs = _to_channels_last(np_mod, imgs, data_format)

    out = _apply_median_filter_batch(np_mod, imgs, kernel_size, padding)

    out = _from_channels_last(np_mod, out, data_format)

    if original_ndim == 3:
        out = out[0]

    return _from_numpy_array(backend_module, out, name, images)
