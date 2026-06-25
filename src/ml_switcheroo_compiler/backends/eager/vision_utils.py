# ruff: noqa: F405, F403

# ruff: noqa: E402
"""Vision utilities."""

from __future__ import annotations


from dataclasses import dataclass


from ml_switcheroo_compiler.ops.configs import PerspectiveConfig
from ml_switcheroo_compiler.backends.eager.utils import (
    _to_channels_last,
    _to_numpy_array,
)


@dataclass
class RandomCropConfig:
    """RandomCropConfig."""

    crop_h: int
    crop_w: int
    b: int
    c: int
    H: int
    W: int
    rng: object


@dataclass
class GeometricGridConfig:
    """GeometricGridConfig."""

    H: int
    W: int
    rng: object
    factor1: object
    factor2: object


@dataclass
class EagerTransformContext:  # pylint: disable=too-many-instance-attributes
    """Class docstring."""

    np_mod: object
    rng: object
    imgs: object
    B: int
    H: int
    W: int
    C: int
    name: str


def _prepare_eager_transform(
    backend_module: object, images: object, seed: object, data_format: object
) -> EagerTransformContext:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        seed: Arg.
        data_format: Arg.
    """
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(seed)
    imgs = _to_channels_last(np_mod, imgs, data_format)
    B, H, W, C = imgs.shape  # type: ignore
    return EagerTransformContext(np_mod, rng, imgs, B, H, W, C, name)  # pragma: no cover


@dataclass
class TransformInterpolationConfig:
    """Class docstring."""

    new_y: object
    new_x: object
    order: int
    fill_value: float


@dataclass
class ResizeContext:
    """Class docstring."""

    H: int
    W: int
    new_H: int
    new_W: int
    align_corners: bool


@dataclass
class MapCoordsContext:
    """MapCoordsContext."""

    np_mod: object
    image: object
    y: object
    x: object
    valid: object


def _map_coords_nearest(ctx: MapCoordsContext) -> object:
    np_mod, image, y, x, valid = ctx.np_mod, ctx.image, ctx.y, ctx.x, ctx.valid
    """Function docstring.

    Args:
        np_mod: Arg.
        image: Arg.
        y: Arg.
        x: Arg.
        valid: Arg.
    """
    y_idx = np_mod.round(y[valid]).astype(np_mod.int32)
    x_idx = np_mod.round(x[valid]).astype(np_mod.int32)
    y_idx = np_mod.clip(y_idx, 0, image.shape[0] - 1)
    x_idx = np_mod.clip(x_idx, 0, image.shape[1] - 1)
    return image[y_idx, x_idx]


def _map_coords_bilinear(ctx: MapCoordsContext) -> object:
    np_mod, image, y, x, valid = ctx.np_mod, ctx.image, ctx.y, ctx.x, ctx.valid
    """Function docstring.

    Args:
        np_mod: Arg.
        image: Arg.
        y: Arg.
        x: Arg.
        valid: Arg.
    """
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
    return image[y0, x0] * w00 + image[y0, x1] * w01 + image[y1, x0] * w10 + image[y1, x1] * w11


def _np_map_coordinates(
    np_mod: object, image: object, coords: object, order: int = 1, fill_value: float = 0.0
) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        image: Arg.
        coords: Arg.
        order: Arg.
        fill_value: Arg.
    """
    y, x = coords[0], coords[1]
    out = np_mod.full(y.shape, fill_value, dtype=image.dtype)
    valid = (y >= 0) & (y <= image.shape[0] - 1) & (x >= 0) & (x <= image.shape[1] - 1)
    if order == 0:
        mctx = MapCoordsContext(np_mod, image, y, x, valid)
        out[valid] = _map_coords_nearest(mctx)
    else:
        mctx = MapCoordsContext(np_mod, image, y, x, valid)
        out[valid] = _map_coords_bilinear(mctx)
    return out


def _compute_perspective_matrix(np_mod: object, src: object, dst: object) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        src: Arg.
        dst: Arg.
    """
    A = np_mod.zeros((*dst.shape[:-2], 8, 8), dtype=np_mod.float32)  # pragma: no cover
    B = np_mod.zeros((*dst.shape[:-2], 8), dtype=np_mod.float32)  # pragma: no cover
    # pragma: no cover
    for i in range(4):  # pragma: no cover
        u = dst[..., i, 0]  # pragma: no cover
        v = dst[..., i, 1]  # pragma: no cover
        x = src[..., i, 0]  # pragma: no cover
        y = src[..., i, 1]  # pragma: no cover
        # pragma: no cover
        A[..., i * 2, 0] = u  # pragma: no cover
        A[..., i * 2, 1] = v  # pragma: no cover
        A[..., i * 2, 2] = 1  # pragma: no cover
        A[..., i * 2, 6] = -x * u  # pragma: no cover
        A[..., i * 2, 7] = -x * v  # pragma: no cover
        # pragma: no cover
        A[..., i * 2 + 1, 3] = u  # pragma: no cover
        A[..., i * 2 + 1, 4] = v  # pragma: no cover
        A[..., i * 2 + 1, 5] = 1  # pragma: no cover
        A[..., i * 2 + 1, 6] = -y * u  # pragma: no cover
        A[..., i * 2 + 1, 7] = -y * v  # pragma: no cover
        # pragma: no cover
        B[..., i * 2] = x  # pragma: no cover
        B[..., i * 2 + 1] = y  # pragma: no cover
    # pragma: no cover
    h = np_mod.linalg.solve(A, B)  # pragma: no cover
    return h  # pragma: no cover


def _generate_perspective_coords(
    np_mod: object, h_batch: object, coords: object
) -> tuple[object, object]:
    """Generate source x and y coordinates for a given batch from homography matrix."""
    H_mat = np_mod.concatenate([h_batch, [1.0]]).reshape(3, 3)  # pragma: no cover
    t_coords = coords @ H_mat.T  # pragma: no cover
    t_coords = t_coords / t_coords[..., 2:3]  # pragma: no cover
    return t_coords[..., 0], t_coords[..., 1]  # pragma: no cover


def _generate_perspective_grid(np_mod: object, H: int, W: int) -> object:
    """Function docstring.

    Args:
        np_mod: Arg.
        H: Arg.
        W: Arg.
    """
    y_grid, x_grid = np_mod.meshgrid(
        np_mod.arange(H), np_mod.arange(W), indexing="ij"
    )  # pragma: no cover
    y_grid = y_grid.astype(np_mod.float32)  # pragma: no cover
    x_grid = x_grid.astype(np_mod.float32)  # pragma: no cover
    ones = np_mod.ones_like(x_grid)  # pragma: no cover
    return np_mod.stack([x_grid, y_grid, ones], axis=-1)  # pragma: no cover


@dataclass
class PerspectiveContext:
    """Class docstring."""

    coords: object
    h: object
    b: int


@dataclass
class PerspectiveChannelContext:
    """PerspectiveChannelContext."""

    np_mod: object
    imgs: object
    out: object
    ctx: PerspectiveContext
    config: PerspectiveConfig


def _apply_perspective_channel(pctx: PerspectiveChannelContext) -> None:
    np_mod, imgs, out, ctx, config = (
        pctx.np_mod,
        pctx.imgs,
        pctx.out,
        pctx.ctx,
        pctx.config,
    )  # pragma: no cover
    """Function docstring.

    Args:
        np_mod: Arg.
        imgs: Arg.
        out: Arg.
        ctx: Arg.
        config: Arg.
    """
    src_x, src_y = _generate_perspective_coords(np_mod, ctx.h, ctx.coords)  # pragma: no cover
    C = imgs.shape[-1]  # pragma: no cover
    order = 1 if config.interpolation == "bilinear" else 0  # pragma: no cover
    for c in range(C):  # pragma: no cover
        channel = imgs[ctx.b, ..., c]  # pragma: no cover
        res = _np_map_coordinates(  # pragma: no cover
            np_mod, channel, [src_y, src_x], order=order, fill_value=config.fill_value
        )
        out[ctx.b, ..., c] = res  # pragma: no cover


def _apply_perspective_batch(
    np_mod: object, imgs: object, h: object, config: PerspectiveConfig
) -> object:
    """Apply perspective transform to a batched image array."""
    B_sz, H, W, C = imgs.shape  # pragma: no cover
    out = np_mod.zeros_like(imgs)  # pragma: no cover
    coords = _generate_perspective_grid(np_mod, H, W)  # pragma: no cover
    # pragma: no cover
    for b in range(B_sz):  # pragma: no branch  # pragma: no cover
        _apply_perspective_channel(
            np_mod, imgs, out, coords, h=h[b], b=b, config=config
        )  # pragma: no cover
    return out  # pragma: no cover


__all__ = [
    "EagerTransformContext",
    "GeometricGridConfig",
    "MapCoordsContext",
    "PerspectiveChannelContext",
    "PerspectiveConfig",
    "PerspectiveContext",
    "RandomCropConfig",
    "ResizeContext",
    "TransformInterpolationConfig",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_apply_perspective_batch",
    "_apply_perspective_channel",
    "_compute_perspective_matrix",
    "_generate_perspective_coords",
    "_generate_perspective_grid",
    "_map_coords_bilinear",
    "_map_coords_nearest",
    "_np_map_coordinates",
    "_prepare_eager_transform",
    "_to_channels_last",
    "_to_numpy_array",
    "annotations",
    "dataclass",
]
