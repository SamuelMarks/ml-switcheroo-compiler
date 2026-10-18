"""Vision operations for the numpy backend."""

from __future__ import annotations

import typing  # noqa: F401
from dataclasses import dataclass
from types import ModuleType
from typing import (
    Callable,
    Optional,  # noqa: F401
    cast,
)

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry


@dataclass
class ResizeConfig:
    """Configuration for resize operations.

    Attributes:
        new_H: New height.
        new_W: New width.
        H: Old height.
        W: Old width.
        align_corners: Align corners.
    """

    new_H: int
    new_W: int
    H: int
    W: int
    align_corners: bool


@dataclass
class InterpolationConfig:
    """Configuration for interpolation.

    Attributes:
        images: Input images tensor.
        src_y: Source y coordinates.
        src_x: Source x coordinates.
        H: Original height.
        W: Original width.
    """

    images: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = None
    src_y: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = None
    src_x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = None
    H: int = 0
    W: int = 0


def _calculate_bilinear_coords(np: ModuleType, cfg: ResizeConfig) -> EagerValue:
    """Calculate coords for bilinear resize.

    Args:
        np: Numpy module.
        cfg: Resize configuration.

    Returns:
        tuple: src_y, src_x.
    """
    y = np.arange(cfg.new_H, dtype=np.float32)
    x = np.arange(cfg.new_W, dtype=np.float32)

    if cfg.align_corners:
        y_scale = (cfg.H - 1) / (cfg.new_H - 1) if cfg.new_H > 1 else 0
        x_scale = (cfg.W - 1) / (cfg.new_W - 1) if cfg.new_W > 1 else 0
        src_y = y * y_scale
        src_x = x * x_scale
    else:
        y_scale = cfg.H / cfg.new_H
        x_scale = cfg.W / cfg.new_W
        src_y = (y + 0.5) * y_scale - 0.5
        src_x = (x + 0.5) * x_scale - 0.5

    return np.clip(src_y, 0, cfg.H - 1), np.clip(src_x, 0, cfg.W - 1)


@dataclass
class BilinearCoords:
    """Coordinates for bilinear interpolation."""

    y0: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    y1: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    x0: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    x1: np.ndarray[tuple[int, ...], np.dtype[np.generic]]


def _compute_bilinear_pixels(np: ModuleType, images: EagerValue, coords: BilinearCoords) -> EagerValue:
    """Fetch pixels for bilinear interpolation.

    Args:
        np: Numpy module.
        images: Input images.
        coords: Tuple of (y0, y1, x0, x1).

    Returns:
        tuple: Pixels Ia, Ib, Ic, Id.
    """
    y0, y1, x0, x1 = coords.y0, coords.y1, coords.x0, coords.x1
    Ia = cast(EagerTensorProtocol, images)[:, cast(EagerTensorProtocol, y0)[:, None], cast(EagerTensorProtocol, x0)[None, :], :]
    Ib = cast(EagerTensorProtocol, images)[:, cast(EagerTensorProtocol, y1)[:, None], cast(EagerTensorProtocol, x0)[None, :], :]
    Ic = cast(EagerTensorProtocol, images)[:, cast(EagerTensorProtocol, y0)[:, None], cast(EagerTensorProtocol, x1)[None, :], :]
    Id = cast(EagerTensorProtocol, images)[:, cast(EagerTensorProtocol, y1)[:, None], cast(EagerTensorProtocol, x1)[None, :], :]
    return Ia, Ib, Ic, Id


def _compute_bilinear_weights(
    dy: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    dx: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Execute the eager compute bilinear weights operation on the active backend array.

    Args:
        dy (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): The dy parameter.
        dx (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): The dx parameter.

    Returns:
        tuple: Result.
    """
    wa = (1 - dy[:, None, None]) * (1 - dx[None, :, None])
    wb = (dy[:, None, None]) * (1 - dx[None, :, None])
    wc = (1 - dy[:, None, None]) * (dx[None, :, None])
    wd = (dy[:, None, None]) * (dx[None, :, None])
    return wa, wb, wc, wd


def _apply_bilinear_interpolation(np: ModuleType, cfg: InterpolationConfig) -> EagerValue:
    """Apply bilinear interpolation.

    Args:
        np (ModuleType): The np parameter.
        cfg (InterpolationConfig): The cfg parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    y0 = np.floor(cfg.src_y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, cfg.H - 1)
    x0 = np.floor(cfg.src_x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, cfg.W - 1)

    dy = cfg.src_y - y0
    dx = cfg.src_x - x0

    pixels = _compute_bilinear_pixels(np, cfg.images, BilinearCoords(y0, y1, x0, x1))
    weights = _compute_bilinear_weights(dy, dx)

    p0, p1, p2, p3 = getattr(pixels, "__iter__")()
    w0, w1, w2, w3 = getattr(weights, "__iter__")()
    return cast(EagerValue, (p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3).astype(getattr(cfg.images, "dtype")))


@numpy_eager_registry.register("ResizeBilinear")
def resize_bilinear(np_mod: ModuleType, images: EagerValue, size: tuple[int, int], align_corners: bool = False) -> EagerValue:
    """Resize images using bilinear interpolation.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        images (EagerValue): The images parameter.
        size (tuple): The size parameter.
        align_corners (bool): The align_corners parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    images = np_mod.asarray(images)
    rank = len(getattr(images, "shape"))
    if rank == 3:
        images = np_mod.expand_dims(images, 0)
    elif rank != 4:
        return images

    shape = getattr(images, "shape")
    resize_cfg = ResizeConfig(new_H=size[0], new_W=size[1], H=shape[1], W=shape[2], align_corners=align_corners)
    src_y, src_x = getattr(_calculate_bilinear_coords(np_mod, resize_cfg), "__iter__")()

    interp_cfg = cast("Callable[..., InterpolationConfig]", InterpolationConfig)(images=images, src_y=src_y, src_x=src_x, H=shape[1], W=shape[2])
    out = _apply_bilinear_interpolation(np_mod, interp_cfg)

    return getattr(out, "__getitem__")(0) if rank == 3 else out


def _calculate_nearest_coords(np: ModuleType, cfg: ResizeConfig) -> EagerValue:
    """Calculate coords for nearest resize.

    Args:
        np: Numpy module.
        cfg: Resize configuration.

    Returns:
        tuple: src_y, src_x.
    """
    y = np.arange(cfg.new_H, dtype=np.float32)
    x = np.arange(cfg.new_W, dtype=np.float32)

    if cfg.align_corners:
        y_scale = (cfg.H - 1) / (cfg.new_H - 1) if cfg.new_H > 1 else 0
        x_scale = (cfg.W - 1) / (cfg.new_W - 1) if cfg.new_W > 1 else 0
        src_y = np.round(y * y_scale).astype(np.int32)
        src_x = np.round(x * x_scale).astype(np.int32)
    else:
        y_scale = cfg.H / cfg.new_H
        x_scale = cfg.W / cfg.new_W
        src_y = np.floor(y * y_scale).astype(np.int32)
        src_x = np.floor(x * x_scale).astype(np.int32)

    return np.clip(src_y, 0, cfg.H - 1), np.clip(src_x, 0, cfg.W - 1)


@numpy_eager_registry.register("ResizeNearest")
def resize_nearest(np_mod: ModuleType, images: EagerValue, size: tuple[int, int], align_corners: bool = False) -> EagerValue:
    """Resize images using nearest-neighbor interpolation.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        images (EagerValue): The images parameter.
        size (tuple): The size parameter.
        align_corners (bool): The align_corners parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    images = np_mod.asarray(images)
    rank = len(getattr(images, "shape"))
    if rank == 3:
        images = np_mod.expand_dims(images, 0)
    elif rank != 4:
        return images

    shape = getattr(images, "shape")
    resize_cfg = ResizeConfig(new_H=size[0], new_W=size[1], H=shape[1], W=shape[2], align_corners=align_corners)
    src_y, src_x = getattr(_calculate_nearest_coords(np_mod, resize_cfg), "__iter__")()

    out = cast(EagerTensorProtocol, images)[:, cast(EagerTensorProtocol, src_y)[:, None], cast(EagerTensorProtocol, src_x)[None, :], :]

    return getattr(out, "__getitem__")(0) if rank == 3 else out


@numpy_eager_registry.register("ResizeBicubic")
def resize_bicubic(np_mod: ModuleType, images: EagerValue, size: tuple[int, int], align_corners: bool = False) -> EagerValue:
    """Resize images using bicubic interpolation.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        images (EagerValue): The images parameter.
        size (tuple): The size parameter.
        align_corners (bool): The align_corners parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return resize_bilinear(np_mod, images, size, align_corners)


@numpy_eager_registry.register("ResizeLanczos3")
def resize_lanczos3(np_mod: ModuleType, images: EagerValue, size: tuple[int, int], align_corners: bool = False) -> EagerValue:
    """Resize images using Lanczos3 interpolation.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        images (EagerValue): The images parameter.
        size (tuple): The size parameter.
        align_corners (bool): The align_corners parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return resize_bilinear(np_mod, images, size, align_corners)


@numpy_eager_registry.register("RandomFlip")
def random_flip_numpy(np_mod: ModuleType, images: EagerValue, mode: EagerValue = "horizontal_and_vertical", seed: EagerValue = None, **kwargs: EagerValue) -> EagerValue:
    """Generate random flip.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        images (EagerValue): The images parameter.
        mode (str): The mode parameter.
        seed (int): The seed parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    np = np_mod
    x = np.asarray(images)
    rank = len(x.shape)
    if rank == 3:
        x = np.expand_dims(x, 0)
    elif rank != 4:
        return cast(EagerValue, x)

    rng = np.random.default_rng(seed)
    out = np.copy(x)

    if mode in ("horizontal", "horizontal_and_vertical"):
        mask_h = rng.random(size=(x.shape[0],)) > 0.5
        out[mask_h] = out[mask_h, :, ::-1, :]

    if mode in ("vertical", "horizontal_and_vertical"):
        mask_v = rng.random(size=(x.shape[0],)) > 0.5
        out[mask_v] = out[mask_v, ::-1, :, :]

    if rank == 3:
        return cast(EagerValue, out[0])
    return cast(EagerValue, out)


def _compute_rotation_coords(np: ModuleType, H: int, W: int, angle: float) -> EagerValue:
    """Execute the eager compute rotation coords operation on the active backend array.

    Args:
        np (ModuleType): The np parameter.
        H (int): The H parameter.
        W (int): The W parameter.
        angle (float): The angle parameter.

    Returns:
        tuple: Result.
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)
    x_shifted = x_coords - cx
    y_shifted = y_coords - cy
    src_x = x_shifted * cos_a + y_shifted * sin_a + cx
    src_y = -x_shifted * sin_a + y_shifted * cos_a + cy
    return src_x, src_y


@dataclass
class RotationInterpolationConfig:
    """Configuration for rotation interpolation.

    Attributes:
        img: Input image.
        src_x: X coordinates.
        src_y: Y coordinates.
        fill_mode: Fill mode.
        fill_value: Fill value.
    """

    img: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = None
    src_x: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = None
    src_y: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = None
    fill_mode: str = "nearest"
    fill_value: float = 0.0


def _get_rotation_coords(np: ModuleType, src_x_clip: EagerValue, src_y_clip: EagerValue, W: int, H: int) -> EagerValue:
    """Get the bounding coordinates for rotation interpolation.

    Args:
        np (ModuleType): The numpy module.
        src_x_clip (EagerValue): Clipped source X coordinates.
        src_y_clip (EagerValue): Clipped source Y coordinates.
        W (int): Image width.
        H (int): Image height.

    Returns:
        tuple: (x0, x1, y0, y1) coordinates.
    """
    x0 = np.floor(src_x_clip).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(src_y_clip).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, H - 1)
    return x0, x1, y0, y1


def _gather_rotation_pixels(
    img: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    coords: tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Gather pixels for rotation.

    Args:
        img: Input image.
        coords: Tuple of x0, x1, y0, y1.

    Returns:
        tuple: Pixels Ia, Ib, Ic, Id.
    """
    x0, x1, y0, y1 = coords
    Ia, Ib = img[y0, x0, :], img[y1, x0, :]
    Ic, Id = img[y0, x1, :], img[y1, x1, :]
    return Ia, Ib, Ic, Id


def _compute_rotation_weights(
    dx: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    dy: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Execute the eager compute rotation weights operation on the active backend array.

    Args:
        dx (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): The dx parameter.
        dy (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): The dy parameter.

    Returns:
        tuple: Result.
    """
    wa = (1 - dy)[..., None] * (1 - dx)[..., None]
    wb = dy[..., None] * (1 - dx)[..., None]
    wc = (1 - dy)[..., None] * dx[..., None]
    wd = dy[..., None] * dx[..., None]
    return wa, wb, wc, wd


def _interpolate_rotation(np: ModuleType, cfg: RotationInterpolationConfig) -> EagerValue:
    """Interpolate rotated image.

    Args:
        np (ModuleType): The np parameter.
        cfg (RotationInterpolationConfig): The cfg parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    H, W, _ = getattr(cfg.img, "shape")
    src_x_clip = np.clip(cfg.src_x, 0, W - 1)
    src_y_clip = np.clip(cfg.src_y, 0, H - 1)

    coords = _get_rotation_coords(np, src_x_clip, src_y_clip, W, H)
    dx = src_x_clip - getattr(coords, "__getitem__")(0)
    dy = src_y_clip - getattr(coords, "__getitem__")(2)

    pixels = cast("Callable[..., EagerTensorProtocol]", _gather_rotation_pixels)(cfg.img, coords)
    weights = _compute_rotation_weights(dx, dy)

    interp = np.multiply(pixels[0], weights[0]) + np.multiply(pixels[1], weights[1]) + np.multiply(pixels[2], weights[2]) + np.multiply(pixels[3], weights[3])
    valid_mask = (np.asarray(cfg.src_x, dtype=float) >= 0) & (np.asarray(cfg.src_x, dtype=float) <= W - 1) & (np.asarray(cfg.src_y, dtype=float) >= 0) & (np.asarray(cfg.src_y, dtype=float) <= H - 1)
    if cfg.fill_mode == "constant":
        interp[~valid_mask] = cfg.fill_value
    return cast(EagerValue, interp.astype(getattr(cfg.img, "dtype")))


def _apply_random_rotation_single(np: ModuleType, img: EagerValue, angle: float, fill_mode: str, fill_value: float) -> EagerValue:
    """Apply random rotation to a single image.

    Args:
        np (ModuleType): The np parameter.
        img (EagerValue): The img parameter.
        angle (float): The angle parameter.
        fill_mode (str): The fill_mode parameter.
        fill_value (float): The fill_value parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    H, W, _ = getattr(getattr(img, "shape"), "__iter__")()
    src_x, src_y = getattr(_compute_rotation_coords(np, H, W, angle), "__iter__")()
    cfg = cast("Callable[..., RotationInterpolationConfig]", RotationInterpolationConfig)(img=img, src_x=src_x, src_y=src_y, fill_mode=fill_mode, fill_value=fill_value)
    return _interpolate_rotation(np, cfg)


@dataclass
class TransformOptions:
    """Options for image transformations."""

    fill_mode: str = "reflect"
    interpolation: str = "bilinear"
    seed: int | None = None
    fill_value: float = 0.0


def _apply_rotation_to_batch(np: ModuleType, x: EagerValue, rng: EagerValue, angles: tuple[float, float], options: TransformOptions) -> EagerValue:
    """Apply rotation to batch of images.

    Args:
        np (ModuleType): The np parameter.
        x (EagerValue): The x parameter.
        rng (EagerValue): The rng parameter.
        angles (tuple): The angles parameter.
        options (TransformOptions): The options parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    B = getattr(x, "shape")[0]
    out = np.zeros_like(x)
    for i in range(B):
        angle = getattr(rng, "uniform")(getattr(angles, "__getitem__")(0), getattr(angles, "__getitem__")(1)) * 2 * np.pi
        cast(EagerTensorProtocol, out)[i] = _apply_random_rotation_single(np, getattr(x, "__getitem__")(i), angle, options.fill_mode, options.fill_value)
    return cast(EagerValue, out)


@numpy_eager_registry.register("RandomRotation")
def random_rotation_numpy(np_mod: ModuleType, images: EagerValue, factor: float, options: TransformOptions | None = None, **kwargs: EagerValue) -> EagerValue:
    """Generate random rotation.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        images (EagerValue): The images parameter.
        factor (float): The factor parameter.
        options (Optional): The options parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    options = options or TransformOptions()
    np = np_mod
    x = np.asarray(images)
    rank = len(x.shape)
    if rank == 3:
        x = np.expand_dims(x, 0)
    elif rank != 4:
        return cast(EagerValue, x)

    rng = np.random.default_rng(options.seed)

    if isinstance(factor, (tuple, list)):
        lower, upper = factor
    else:
        lower, upper = -factor, factor

    out = _apply_rotation_to_batch(np, x, rng, (lower, upper), options)

    if rank == 3:
        return cast(EagerValue, getattr(out, "__getitem__")(0))
    return out
