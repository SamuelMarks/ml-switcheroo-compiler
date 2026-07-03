"""Vision utilities."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from ml_switcheroo_compiler.backends.eager.utils import (
    _from_channels_last,
    _from_numpy_array,
    _to_channels_last,
    _to_numpy_array,
)
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3, MAGIC_VAL_4
from ml_switcheroo_compiler.ops.configs import ElasticConfig, PerspectiveConfig, ResizeOptions

from .vision_utils import (
    ResizeContext,
    TransformInterpolationConfig,
    _apply_perspective_batch,
    _compute_perspective_matrix,
    _np_map_coordinates,
)


def perspective_transform_eager(
    backend_module: object,
    images: object,
    start_points: object,
    end_points: object,
    config: PerspectiveConfig,
) -> object:
    """Evaluate perspective transform eagerly."""
    name = getattr(backend_module, "__name__", "")
    config.interpolation = getattr(config, "interpolation", "bilinear")
    config.fill_value = getattr(config, "fill_value", 0.0)
    config.data_format = getattr(config, "data_format", None)
    if name == "keras.ops":
        return backend_module.image.perspective_transform(
            images,
            start_points,
            end_points,
            config.interpolation,
            config.fill_value,
            config.data_format,
        )
    np_mod = __import__("numpy")
    imgs = _to_numpy_array(np_mod, images, name)
    src = _to_numpy_array(np_mod, start_points, name)
    dst = _to_numpy_array(np_mod, end_points, name)
    imgs = _to_channels_last(np_mod, imgs, config.data_format)
    h = _compute_perspective_matrix(np_mod, src, dst)
    has_batch = imgs.ndim == MAGIC_VAL_4
    if not has_batch:
        imgs = imgs[None, ...]
        h = h[None, ...]
    out = _apply_perspective_batch(np_mod, imgs, h, config)
    if not has_batch:
        out = out[0]
    out = _from_channels_last(np_mod, out, config.data_format)
    return _from_numpy_array(backend_module, out, name, images)


def _apply_elastic_batch(np_mod: object, imgs: object, config: TransformInterpolationConfig) -> object:
    """Apply elastic coordinates across a batch."""
    (B, H, W, C) = imgs.shape
    out = np_mod.zeros_like(imgs)
    for b in range(B):
        for c in range(C):
            coords = [config.new_y[b], config.new_x[b]]
            out[b, ..., c] = _np_map_coordinates(np_mod, imgs[b, ..., c], coords, order=config.order, fill_value=config.fill_value)
    return out


@dataclass
class ElasticGridContext:
    """ElasticGridContext."""

    np_mod: object
    H: int
    W: int
    B: int
    disp: object


def _compute_elastic_grid(ctx: ElasticGridContext) -> tuple[object, object]:
    """Function docstring."""
    (np_mod, H, W, B, disp) = (ctx.np_mod, ctx.H, ctx.W, ctx.B, ctx.disp)
    "Function docstring.\n\n    Args:\n        np_mod: Arg.\n        H: Arg.\n        W: Arg.\n        B: Arg.\n        disp: Arg.\n    "
    (y, x) = np_mod.meshgrid(np_mod.arange(H), np_mod.arange(W), indexing="ij")
    y = np_mod.broadcast_to(y, (B, H, W))
    x = np_mod.broadcast_to(x, (B, H, W))
    dy = disp[..., 0]
    dx = disp[..., 1]
    return (y + dy, x + dx)


def elastic_transform_eager(backend_module: object, images: object, displacement: object, config: ElasticConfig) -> object:
    """Evaluate elastic transform eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")
    imgs = _to_numpy_array(np_mod, images, name)
    disp = _to_numpy_array(np_mod, displacement, name)
    original_ndim = imgs.ndim
    if original_ndim == MAGIC_VAL_3:
        imgs = imgs[None, ...]
        disp = disp[None, ...]
    imgs = _to_channels_last(np_mod, imgs, config.data_format)
    (B, H, W, C) = imgs.shape
    (new_y, new_x) = _compute_elastic_grid(ElasticGridContext(np_mod, H, W, B, disp))
    order = 1 if config.interpolation == "bilinear" else 0
    config = TransformInterpolationConfig(new_y=new_y, new_x=new_x, order=order, fill_value=config.fill_value)
    out = _apply_elastic_batch(np_mod, imgs, config)
    out = _from_channels_last(np_mod, out, config.data_format)
    if original_ndim == MAGIC_VAL_3:
        out = out[0]
    return _from_numpy_array(backend_module, out, name, images)


def _get_resize_interpolation_order(interpolation: str) -> int:
    """Get scipy ndimage order for interpolation string."""
    if interpolation == "nearest":
        return 0
    elif interpolation in ("bicubic", "lanczos3"):
        return 3
    return 1


def _compute_resize_grid(np_mod: object, ctx: ResizeContext) -> tuple[object, object]:
    """Compute the sampling grid for resize operation."""
    (H, W, new_H, new_W, align_corners) = (ctx.H, ctx.W, ctx.new_H, ctx.new_W, ctx.align_corners)
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


def _apply_resize_batch(np_mod: object, imgs: object, out: object, coords: tuple[object, object], order: int) -> None:
    """Function docstring.

    Args:
        np_mod: Arg.
        imgs: Arg.
        out: Arg.
        coords: Arg.
        order: Arg.
    """
    B = imgs.shape[0]
    C = imgs.shape[-1]
    (yy, xx) = coords
    for b in range(B):
        for c in range(C):
            out[b, ..., c] = _np_map_coordinates(np_mod, imgs[b, ..., c], [yy, xx], order=order)


def resize_eager(backend_module: object, images: object, size: tuple[int, int], config: ResizeOptions) -> object:
    """Evaluate resize eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")
    if name == "keras.ops":
        images_tf = tf.convert_to_tensor(images)
        res = tf.image.resize(images_tf, size, method=config.interpolation, antialias=True)
        return backend_module.convert_to_tensor(res)
    imgs = _to_numpy_array(np_mod, images, name)
    (B, H, W, C) = imgs.shape
    (new_H, new_W) = size
    order = _get_resize_interpolation_order(config.interpolation)
    out = np_mod.zeros((B, new_H, new_W, C), dtype=np_mod.float32)
    ctx = ResizeContext(H=H, W=W, new_H=new_H, new_W=new_W, align_corners=config.align_corners)
    (yy, xx) = _compute_resize_grid(np_mod, ctx)
    _apply_resize_batch(np_mod, imgs, out, (yy, xx), order)
    return _from_numpy_array(backend_module, out, name, images)
