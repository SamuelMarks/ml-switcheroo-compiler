"""Signal utilities."""

import typing


def _to_numpy_array(np_mod: object, x: object, name: str) -> object:
    if name == "torch":
        return x.detach().cpu().numpy()
    if name == "mlx.core":
        return np_mod.array(x)
    if hasattr(x, "numpy"):
        return x.numpy()
    return np_mod.asarray(x)


def _from_numpy_array(
    backend_module: object, out: object, name: str, original_tensor: object = None
) -> object:
    if name == "torch":
        import torch

        return (
            torch.tensor(out, dtype=original_tensor.dtype, device=original_tensor.device)
            if original_tensor is not None
            else torch.tensor(out)
        )
    if name == "mlx.core":
        import mlx.core as mx

        return (
            mx.array(out, dtype=original_tensor.dtype)
            if original_tensor is not None
            else mx.array(out)
        )
    if name == "jax.numpy":
        import jax.numpy as jnp

        return (
            jnp.array(out, dtype=original_tensor.dtype)
            if original_tensor is not None
            else jnp.array(out)
        )
    if name == "keras.ops":
        return backend_module.convert_to_tensor(
            out, dtype=original_tensor.dtype if original_tensor is not None else "float32"
        )
    return (
        backend_module.array(out, dtype=original_tensor.dtype)
        if original_tensor is not None
        else backend_module.array(out)
    )


def _to_channels_last(np_mod: object, imgs: object, data_format: typing.Optional[str]) -> object:
    if data_format == "channels_first" and imgs.ndim >= 3:
        if imgs.ndim == 4:
            return np_mod.transpose(imgs, (0, 2, 3, 1))
        elif imgs.ndim == 3:
            return np_mod.transpose(imgs, (1, 2, 0))
    return imgs


def _from_channels_last(np_mod: object, out: object, data_format: typing.Optional[str]) -> object:
    if data_format == "channels_first" and out.ndim >= 3:
        if out.ndim == 4:
            return np_mod.transpose(out, (0, 3, 1, 2))
        elif out.ndim == 3:
            return np_mod.transpose(out, (2, 0, 1))
    return out


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


def gaussian_blur_eager(
    backend_module: object,
    images: object,
    config_obj: typing.Optional[object] = None,
    **kwargs: object,
) -> object:
    """Evaluate gaussian blur eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import BlurConfig

        config_obj = BlurConfig(
            kernel_size=kwargs.get("kernel_size", (3, 3)),
            sigma=kwargs.get("sigma", (1.0, 1.0)),
            data_format=kwargs.get("data_format", None),
        )
    kernel_size = config_obj.kernel_size
    sigma = config_obj.sigma
    padding = kwargs.get("padding", "same")
    data_format = config_obj.data_format

    imgs = _to_numpy_array(np_mod, images, name)
    kernel = _generate_gaussian_kernel(np_mod, kernel_size, sigma)

    original_ndim = imgs.ndim
    if original_ndim == 3:
        imgs = imgs[None, ...]

    imgs = _to_channels_last(np_mod, imgs, data_format)

    mode = "same" if padding == "same" else "valid"
    out = _apply_conv2d_batch(np_mod, imgs, kernel, mode)

    out = _from_channels_last(np_mod, out, data_format)

    if original_ndim == 3:
        out = out[0]

    return _from_numpy_array(backend_module, out, name, images)


def _apply_median_filter_batch(
    np_mod: object, imgs: object, kernel_size: tuple[int, int], padding: str
) -> object:
    import scipy.ndimage

    B, H, W, C = imgs.shape
    ky, kx = kernel_size

    if padding == "valid":
        out = np_mod.zeros((B, H - ky + 1, W - kx + 1, C), dtype=imgs.dtype)
    else:
        out = np_mod.zeros_like(imgs)

    for b in range(B):
        for c in range(C):
            filtered = scipy.ndimage.median_filter(
                imgs[b, ..., c], size=(ky, kx), mode="constant", cval=0.0
            )
            if padding == "valid":
                pad_y_top = ky // 2
                pad_x_left = kx // 2
                filtered = filtered[
                    pad_y_top : pad_y_top + out.shape[1], pad_x_left : pad_x_left + out.shape[2]
                ]
            out[b, ..., c] = filtered
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
