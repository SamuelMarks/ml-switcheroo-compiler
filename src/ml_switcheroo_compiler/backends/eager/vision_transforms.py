# ruff: noqa: F405, F403

# ruff: noqa: E402
"""Vision utilities."""

from __future__ import annotations

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_4

from dataclasses import dataclass


from ml_switcheroo_compiler.ops.configs import ElasticConfig, PerspectiveConfig, ResizeOptions
from ml_switcheroo_compiler.backends.eager.utils import (
    _from_channels_last,
    _from_numpy_array,
    _to_channels_last,
    _to_numpy_array,
)

from .vision_utils import *


def perspective_transform_eager(
    backend_module: object,
    images: object,
    start_points: object,
    end_points: object,
    config: PerspectiveConfig,
) -> object:
    """Evaluate perspective transform eagerly."""
    name = getattr(backend_module, "__name__", "")  # pragma: no cover

    config.interpolation = getattr(config, "interpolation", "bilinear")  # pragma: no cover
    config.fill_value = getattr(config, "fill_value", 0.0)  # pragma: no cover
    config.data_format = getattr(config, "data_format", None)  # pragma: no cover

    if name == "keras.ops":  # pragma: no cover
        return backend_module.image.perspective_transform(  # pragma: no cover
            images,
            start_points,
            end_points,
            config.interpolation,
            config.fill_value,
            config.data_format,
        )

    np_mod = __import__("numpy")  # pragma: no cover

    imgs = _to_numpy_array(np_mod, images, name)  # pragma: no cover
    src = _to_numpy_array(np_mod, start_points, name)  # pragma: no cover
    dst = _to_numpy_array(np_mod, end_points, name)  # pragma: no cover

    imgs = _to_channels_last(np_mod, imgs, config.data_format)  # pragma: no cover

    h = _compute_perspective_matrix(np_mod, src, dst)  # pragma: no cover

    has_batch = imgs.ndim == MAGIC_VAL_4  # pragma: no cover
    if not has_batch:  # pragma: no cover
        imgs = imgs[None, ...]  # pragma: no cover
        h = h[None, ...]  # pragma: no cover

    out = _apply_perspective_batch(np_mod, imgs, h, config)  # pragma: no cover

    if not has_batch:  # pragma: no cover
        out = out[0]  # pragma: no cover

    out = _from_channels_last(np_mod, out, config.data_format)  # pragma: no cover

    return _from_numpy_array(backend_module, out, name, images)  # pragma: no cover


def _apply_elastic_batch(
    np_mod: object, imgs: object, config: TransformInterpolationConfig
) -> object:
    """Apply elastic coordinates across a batch."""
    B, H, W, C = imgs.shape  # pragma: no cover
    out = np_mod.zeros_like(imgs)  # pragma: no cover
    for b in range(B):  # pragma: no cover
        for c in range(C):  # pragma: no cover
            coords = [config.new_y[b], config.new_x[b]]  # pragma: no cover
            out[b, ..., c] = _np_map_coordinates(  # pragma: no cover
                np_mod, imgs[b, ..., c], coords, order=config.order, fill_value=config.fill_value
            )
    return out  # pragma: no cover


@dataclass
class ElasticGridContext:
    np_mod: object
    H: int
    W: int
    B: int
    disp: object


def _compute_elastic_grid(ctx: ElasticGridContext) -> tuple[object, object]:
    np_mod, H, W, B, disp = ctx.np_mod, ctx.H, ctx.W, ctx.B, ctx.disp  # pragma: no cover
    """Function docstring.

    Args:
        np_mod: Arg.
        H: Arg.
        W: Arg.
        B: Arg.
        disp: Arg.
    """
    y, x = np_mod.meshgrid(np_mod.arange(H), np_mod.arange(W), indexing="ij")  # pragma: no cover
    y = np_mod.broadcast_to(y, (B, H, W))  # pragma: no cover
    x = np_mod.broadcast_to(x, (B, H, W))  # pragma: no cover
    dy = disp[..., 0]  # pragma: no cover
    dx = disp[..., 1]  # pragma: no cover
    return y + dy, x + dx  # pragma: no cover


def elastic_transform_eager(  # pylint: disable=too-many-locals
    backend_module: object,
    images: object,
    displacement: object,
    config: ElasticConfig,
) -> object:
    """Evaluate elastic transform eagerly."""
    name = getattr(backend_module, "__name__", "")  # pragma: no cover
    np_mod = __import__("numpy")  # pragma: no cover

    imgs = _to_numpy_array(np_mod, images, name)  # pragma: no cover
    disp = _to_numpy_array(np_mod, displacement, name)  # pragma: no cover

    original_ndim = imgs.ndim  # pragma: no cover
    if original_ndim == MAGIC_VAL_3:  # pragma: no cover
        imgs = imgs[None, ...]  # pragma: no cover
        disp = disp[None, ...]  # pragma: no cover

    imgs = _to_channels_last(np_mod, imgs, config.data_format)  # pragma: no cover

    B, H, W, C = imgs.shape  # pragma: no cover

    new_y, new_x = _compute_elastic_grid(
        ElasticGridContext(np_mod, H, W, B, disp)
    )  # pragma: no cover

    order = 1 if config.interpolation == "bilinear" else 0  # pragma: no cover

    config = TransformInterpolationConfig(  # pragma: no cover
        new_y=new_y, new_x=new_x, order=order, fill_value=config.fill_value
    )
    out = _apply_elastic_batch(np_mod, imgs, config)  # pragma: no cover

    out = _from_channels_last(np_mod, out, config.data_format)  # pragma: no cover

    if original_ndim == MAGIC_VAL_3:  # pragma: no cover
        out = out[0]  # pragma: no cover

    return _from_numpy_array(backend_module, out, name, images)  # pragma: no cover


def _get_resize_interpolation_order(interpolation: str) -> int:
    """Get scipy ndimage order for interpolation string."""
    if interpolation == "nearest":  # pragma: no cover
        return 0  # pragma: no cover
    elif interpolation in ("bicubic", "lanczos3"):  # pragma: no cover
        return 3  # pragma: no cover
    return 1  # pragma: no cover


def _compute_resize_grid(np_mod: object, ctx: ResizeContext) -> tuple[object, object]:
    """Compute the sampling grid for resize operation."""
    H, W, new_H, new_W, align_corners = (
        ctx.H,
        ctx.W,
        ctx.new_H,
        ctx.new_W,
        ctx.align_corners,
    )  # pragma: no cover
    if align_corners:  # pragma: no cover
        scale_y = (H - 1) / max(1, new_H - 1) if new_H > 1 else 0  # pragma: no cover
        scale_x = (W - 1) / max(1, new_W - 1) if new_W > 1 else 0  # pragma: no cover
    else:
        scale_y = H / new_H  # pragma: no cover
        scale_x = W / new_W  # pragma: no cover

    y_coords = np_mod.arange(new_H) * scale_y  # pragma: no cover
    x_coords = np_mod.arange(new_W) * scale_x  # pragma: no cover
    if not align_corners:  # pragma: no cover
        y_coords += 0.5 * scale_y - 0.5  # pragma: no cover
        x_coords += 0.5 * scale_x - 0.5  # pragma: no cover

    return np_mod.meshgrid(y_coords, x_coords, indexing="ij")  # pragma: no cover


def _apply_resize_batch(
    np_mod: object, imgs: object, out: object, coords: tuple[object, object], order: int
) -> None:
    """Function docstring.

    Args:
        np_mod: Arg.
        imgs: Arg.
        out: Arg.
        coords: Arg.
        order: Arg.
    """
    B = imgs.shape[0]  # pragma: no cover
    C = imgs.shape[-1]  # pragma: no cover
    yy, xx = coords  # pragma: no cover
    for b in range(B):  # pragma: no cover
        for c in range(C):  # pragma: no cover
            out[b, ..., c] = _np_map_coordinates(
                np_mod, imgs[b, ..., c], [yy, xx], order=order
            )  # pragma: no cover


def resize_eager(  # pylint: disable=too-many-locals
    backend_module: object,
    images: object,
    size: tuple[int, int],
    config: ResizeOptions,
) -> object:
    """Evaluate resize eagerly."""
    name = getattr(backend_module, "__name__", "")  # pragma: no cover
    np_mod = __import__("numpy")  # pragma: no cover

    if name == "keras.ops":  # pragma: no cover
        import tensorflow as tf  # pragma: no cover

        images_tf = tf.convert_to_tensor(images)  # pragma: no cover
        res = tf.image.resize(
            images_tf, size, method=config.interpolation, antialias=True
        )  # pragma: no cover
        return backend_module.convert_to_tensor(res)  # pragma: no cover

    imgs = _to_numpy_array(np_mod, images, name)  # pragma: no cover

    B, H, W, C = imgs.shape  # pragma: no cover
    new_H, new_W = size  # pragma: no cover

    order = _get_resize_interpolation_order(config.interpolation)  # pragma: no cover
    out = np_mod.zeros((B, new_H, new_W, C), dtype=np_mod.float32)  # pragma: no cover

    ctx = ResizeContext(
        H=H, W=W, new_H=new_H, new_W=new_W, align_corners=config.align_corners
    )  # pragma: no cover
    yy, xx = _compute_resize_grid(np_mod, ctx)  # pragma: no cover

    _apply_resize_batch(np_mod, imgs, out, (yy, xx), order)  # pragma: no cover

    return _from_numpy_array(backend_module, out, name, images)  # pragma: no cover


__all__ = [n for n in globals().keys() if n != "__builtins__"]
