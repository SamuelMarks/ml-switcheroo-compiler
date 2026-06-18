"""Vision utilities."""

import typing
from ml_switcheroo_compiler.ops.configs import PerspectiveConfig


def _to_numpy_array(np_mod: object, x: object, name: str) -> object:
    """Convert tensor to numpy array."""
    if name == "torch":
        return x.detach().cpu().numpy()
    if name == "mlx.core":
        return np_mod.array(x)
    if hasattr(x, "numpy"):
        return x.numpy()
    return np_mod.asarray(x)


def _from_numpy_array(
    backend_module: object, out: object, name: str, original_image: object = None
) -> object:
    """Convert numpy array back to backend tensor."""
    if name == "torch":
        import torch

        if original_image is not None:
            return torch.tensor(out, dtype=original_image.dtype, device=original_image.device)
        return torch.tensor(out)
    if name == "mlx.core":
        import mlx.core as mx

        if original_image is not None:
            return mx.array(out, dtype=original_image.dtype)
        return mx.array(out)
    if name == "jax.numpy":
        import jax.numpy as jnp

        if original_image is not None:
            return jnp.array(out, dtype=original_image.dtype)
        return jnp.array(out)

    if original_image is not None:
        return backend_module.array(out, dtype=original_image.dtype)
    return backend_module.array(out)


def _np_map_coordinates(
    np_mod: object, image: object, coords: object, order: int = 1, fill_value: float = 0.0
) -> object:
    y, x = coords[0], coords[1]
    out = np_mod.full(y.shape, fill_value, dtype=image.dtype)
    valid = (y >= 0) & (y <= image.shape[0] - 1) & (x >= 0) & (x <= image.shape[1] - 1)
    if order == 0:
        y_idx = np_mod.round(y[valid]).astype(np_mod.int32)
        x_idx = np_mod.round(x[valid]).astype(np_mod.int32)
        y_idx = np_mod.clip(y_idx, 0, image.shape[0] - 1)
        x_idx = np_mod.clip(x_idx, 0, image.shape[1] - 1)
        out[valid] = image[y_idx, x_idx]
    else:
        y0 = np_mod.floor(y[valid]).astype(np_mod.int32)
        x0 = np_mod.floor(x[valid]).astype(np_mod.int32)
        y1 = y0 + 1
        x1 = x0 + 1
        y0 = np_mod.clip(y0, 0, image.shape[0] - 1)
        x0 = np_mod.clip(x0, 0, image.shape[1] - 1)
        y1 = np_mod.clip(y1, 0, image.shape[0] - 1)
        x1 = np_mod.clip(x1, 0, image.shape[1] - 1)
        dy = y[valid] - y0
        dx = x[valid] - x0
        w00 = (1 - dy) * (1 - dx)
        w01 = (1 - dy) * dx
        w10 = dy * (1 - dx)
        w11 = dy * dx
        val = image[y0, x0] * w00 + image[y0, x1] * w01 + image[y1, x0] * w10 + image[y1, x1] * w11
        out[valid] = val
    return out


def _to_channels_last(np_mod: object, imgs: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_first to channels_last if needed."""
    if data_format == "channels_first" and imgs.ndim >= 3:
        if imgs.ndim == 4:
            return np_mod.transpose(imgs, (0, 2, 3, 1))
        elif imgs.ndim == 3:
            return np_mod.transpose(imgs, (1, 2, 0))
    return imgs


def _from_channels_last(np_mod: object, out: object, data_format: typing.Optional[str]) -> object:
    """Transpose images from channels_last to channels_first if needed."""
    if data_format == "channels_first" and out.ndim >= 3:
        if out.ndim == 4:
            return np_mod.transpose(out, (0, 3, 1, 2))
        elif out.ndim == 3:
            return np_mod.transpose(out, (2, 0, 1))
    return out


def _compute_perspective_matrix(np_mod: object, src: object, dst: object) -> object:
    A = np_mod.zeros((*dst.shape[:-2], 8, 8), dtype=np_mod.float32)
    B = np_mod.zeros((*dst.shape[:-2], 8), dtype=np_mod.float32)

    for i in range(4):
        u = dst[..., i, 0]
        v = dst[..., i, 1]
        x = src[..., i, 0]
        y = src[..., i, 1]

        A[..., i * 2, 0] = u
        A[..., i * 2, 1] = v
        A[..., i * 2, 2] = 1
        A[..., i * 2, 6] = -x * u
        A[..., i * 2, 7] = -x * v

        A[..., i * 2 + 1, 3] = u
        A[..., i * 2 + 1, 4] = v
        A[..., i * 2 + 1, 5] = 1
        A[..., i * 2 + 1, 6] = -y * u
        A[..., i * 2 + 1, 7] = -y * v

        B[..., i * 2] = x
        B[..., i * 2 + 1] = y

    h = np_mod.linalg.solve(A, B)
    return h


def _generate_perspective_coords(
    np_mod: object, h_batch: object, coords: object
) -> tuple[object, object]:
    """Generate source x and y coordinates for a given batch from homography matrix."""
    H_mat = np_mod.concatenate([h_batch, [1.0]]).reshape(3, 3)
    t_coords = coords @ H_mat.T
    t_coords = t_coords / t_coords[..., 2:3]
    return t_coords[..., 0], t_coords[..., 1]


def _apply_perspective_batch(
    np_mod: object, imgs: object, h: object, order: int, fill_value: float
) -> object:
    """Apply perspective transform to a batched image array."""
    B_sz, H, W, C = imgs.shape
    out = np_mod.zeros_like(imgs)
    y_grid, x_grid = np_mod.meshgrid(np_mod.arange(H), np_mod.arange(W), indexing="ij")
    y_grid = y_grid.astype(np_mod.float32)
    x_grid = x_grid.astype(np_mod.float32)
    ones = np_mod.ones_like(x_grid)
    coords = np_mod.stack([x_grid, y_grid, ones], axis=-1)

    for b in range(B_sz):
        src_x, src_y = _generate_perspective_coords(np_mod, h[b], coords)
        for c in range(C):
            channel = imgs[b, ..., c]
            res = _np_map_coordinates(
                np_mod, channel, [src_y, src_x], order=order, fill_value=fill_value
            )
            out[b, ..., c] = res
    return out


def perspective_transform_eager(
    backend_module: object,
    images: object,
    start_points: object,
    end_points: object,
    config: PerspectiveConfig,
) -> object:
    """Evaluate perspective transform eagerly."""
    name = getattr(backend_module, "__name__", "")

    interpolation = config.get("interpolation", "bilinear")
    fill_value = config.get("fill_value", 0.0)
    data_format = config.get("data_format", None)

    if name == "keras.ops":
        return backend_module.image.perspective_transform(
            images, start_points, end_points, interpolation, fill_value, data_format
        )

    np_mod = __import__("numpy")

    imgs = _to_numpy_array(np_mod, images, name)
    src = _to_numpy_array(np_mod, start_points, name)
    dst = _to_numpy_array(np_mod, end_points, name)

    imgs = _to_channels_last(np_mod, imgs, data_format)

    h = _compute_perspective_matrix(np_mod, src, dst)

    has_batch = imgs.ndim == 4
    if not has_batch:
        imgs = imgs[None, ...]
        h = h[None, ...]

    order = 1 if interpolation == "bilinear" else 0
    out = _apply_perspective_batch(np_mod, imgs, h, order, fill_value)

    if not has_batch:
        out = out[0]

    out = _from_channels_last(np_mod, out, data_format)

    return _from_numpy_array(backend_module, out, name, images)


def _apply_elastic_batch(
    np_mod: object, imgs: object, new_y: object, new_x: object, order: int, fill_value: float
) -> object:
    """Apply elastic coordinates across a batch."""
    B, H, W, C = imgs.shape
    out = np_mod.zeros_like(imgs)
    for b in range(B):
        for c in range(C):
            coords = [new_y[b], new_x[b]]
            out[b, ..., c] = _np_map_coordinates(
                np_mod, imgs[b, ..., c], coords, order=order, fill_value=fill_value
            )
    return out


def elastic_transform_eager(
    backend_module: object,
    images: object,
    displacement: object,
    interpolation: str = "bilinear",
    fill_value: float = 0.0,
    data_format: typing.Optional[str] = None,
) -> object:
    """Evaluate elastic transform eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    imgs = _to_numpy_array(np_mod, images, name)
    disp = _to_numpy_array(np_mod, displacement, name)

    original_ndim = imgs.ndim
    if original_ndim == 3:
        imgs = imgs[None, ...]
        disp = disp[None, ...]

    imgs = _to_channels_last(np_mod, imgs, data_format)

    B, H, W, C = imgs.shape

    y, x = np_mod.meshgrid(np_mod.arange(H), np_mod.arange(W), indexing="ij")
    y = np_mod.broadcast_to(y, (B, H, W))
    x = np_mod.broadcast_to(x, (B, H, W))

    dy = disp[..., 0]
    dx = disp[..., 1]

    new_y = y + dy
    new_x = x + dx

    order = 1 if interpolation == "bilinear" else 0

    out = _apply_elastic_batch(np_mod, imgs, new_y, new_x, order, fill_value)

    out = _from_channels_last(np_mod, out, data_format)

    if original_ndim == 3:
        out = out[0]

    return _from_numpy_array(backend_module, out, name, images)


def _get_resize_interpolation_order(interpolation: str) -> int:
    """Get scipy ndimage order for interpolation string."""
    if interpolation == "nearest":
        return 0
    elif interpolation in ("bicubic", "lanczos3"):
        return 3
    return 1


def _compute_resize_grid(
    np_mod: object, H: int, W: int, new_H: int, new_W: int, align_corners: bool
) -> tuple[object, object]:
    """Compute the sampling grid for resize operation."""
    if align_corners:
        scale_y = (H - 1) / max(1, new_H - 1) if new_H > 1 else 0
        scale_x = (W - 1) / max(1, new_W - 1) if new_W > 1 else 0
    else:
        scale_y = H / new_H
        scale_x = W / new_W

    y_coords = np_mod.arange(new_H) * scale_y
    x_coords = np_mod.arange(new_W) * scale_x
    if not align_corners:
        y_coords += 0.5 * scale_y - 0.5
        x_coords += 0.5 * scale_x - 0.5

    return np_mod.meshgrid(y_coords, x_coords, indexing="ij")


def resize_eager(
    backend_module: object,
    images: object,
    size: tuple[int, int],
    interpolation: str,
    align_corners: bool = False,
) -> object:
    """Evaluate resize eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    if name == "keras.ops":
        import tensorflow as tf

        images_tf = tf.convert_to_tensor(images)
        res = tf.image.resize(images_tf, size, method=interpolation, antialias=True)
        return backend_module.convert_to_tensor(res)

    imgs = _to_numpy_array(np_mod, images, name)
    import scipy.ndimage

    B, H, W, C = imgs.shape
    new_H, new_W = size

    order = _get_resize_interpolation_order(interpolation)
    out = np_mod.zeros((B, new_H, new_W, C), dtype=np_mod.float32)

    yy, xx = _compute_resize_grid(np_mod, H, W, new_H, new_W, align_corners)

    for b in range(B):
        for c in range(C):
            channel = imgs[b, ..., c]
            out[b, ..., c] = scipy.ndimage.map_coordinates(
                channel, [yy, xx], order=order, mode="nearest"
            )

    return _from_numpy_array(backend_module, out, name, images)
