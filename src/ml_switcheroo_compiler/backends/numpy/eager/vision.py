# ruff: noqa: E501
"""Vision operations for the numpy backend."""

from dataclasses import dataclass
from typing import Optional

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


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

    images: object
    src_y: object
    src_x: object
    H: int
    W: int


def _calculate_bilinear_coords(np: object, cfg: ResizeConfig) -> tuple:
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

    y0: object
    y1: object
    x0: object
    x1: object


def _compute_bilinear_pixels(np: object, images: object, coords: BilinearCoords) -> tuple:
    """Fetch pixels for bilinear interpolation.

    Args:
        np: Numpy module.
        images: Input images.
        coords: Tuple of (y0, y1, x0, x1).

    Returns:
        tuple: Pixels Ia, Ib, Ic, Id.
    """
    y0, y1, x0, x1 = coords.y0, coords.y1, coords.x0, coords.x1
    Ia = images[:, y0[:, None], x0[None, :], :]
    Ib = images[:, y1[:, None], x0[None, :], :]
    Ic = images[:, y0[:, None], x1[None, :], :]
    Id = images[:, y1[:, None], x1[None, :], :]
    return Ia, Ib, Ic, Id


def _compute_bilinear_weights(dy: object, dx: object) -> tuple:
    """Compute weights for bilinear interpolation.

    Args:
        dy: Y difference.
        dx: X difference.

    Returns:
        tuple: Weights wa, wb, wc, wd.
    """
    wa = (1 - dy[:, None, None]) * (1 - dx[None, :, None])
    wb = (dy[:, None, None]) * (1 - dx[None, :, None])
    wc = (1 - dy[:, None, None]) * (dx[None, :, None])
    wd = (dy[:, None, None]) * (dx[None, :, None])
    return wa, wb, wc, wd


def _apply_bilinear_interpolation(np: object, cfg: InterpolationConfig) -> object:
    """Apply bilinear interpolation.

    Args:
        np: Numpy module.
        cfg: Interpolation configuration.

    Returns:
        Any: Resized images.
    """
    y0 = np.floor(cfg.src_y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, cfg.H - 1)
    x0 = np.floor(cfg.src_x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, cfg.W - 1)

    dy = cfg.src_y - y0
    dx = cfg.src_x - x0

    pixels = _compute_bilinear_pixels(np, cfg.images, BilinearCoords(y0, y1, x0, x1))
    weights = _compute_bilinear_weights(dy, dx)

    return (pixels[0] * weights[0] + pixels[1] * weights[1] + pixels[2] * weights[2] + pixels[3] * weights[3]).astype(cfg.images.dtype)


@numpy_eager_registry.register("ResizeBilinear")
def resize_bilinear(np_mod: object, images: object, size: tuple[int, int], align_corners: bool = False) -> object:
    """Resize images using bilinear interpolation.

    Args:
        np_mod: Numpy module.
        images: The images to resize.
        size: The target size (height, width).
        align_corners: Whether to align corners.

    Returns:
        Any: The resized images.
    """
    images = np_mod.asarray(images)
    rank = len(images.shape)
    if rank == 3:
        images = np_mod.expand_dims(images, 0)
    elif rank != 4:
        return images

    shape = images.shape
    resize_cfg = ResizeConfig(new_H=size[0], new_W=size[1], H=shape[1], W=shape[2], align_corners=align_corners)
    src_y, src_x = _calculate_bilinear_coords(np_mod, resize_cfg)

    interp_cfg = InterpolationConfig(images=images, src_y=src_y, src_x=src_x, H=shape[1], W=shape[2])
    out = _apply_bilinear_interpolation(np_mod, interp_cfg)

    return out[0] if rank == 3 else out


def _calculate_nearest_coords(np: object, cfg: ResizeConfig) -> tuple:
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
def resize_nearest(np_mod: object, images: object, size: tuple[int, int], align_corners: bool = False) -> object:
    """Resize images using nearest-neighbor interpolation.

    Args:
        np_mod: Numpy module.
        images: The images to resize.
        size: The target size (height, width).
        align_corners: Whether to align corners.

    Returns:
        Any: The resized images.
    """
    images = np_mod.asarray(images)
    rank = len(images.shape)
    if rank == 3:
        images = np_mod.expand_dims(images, 0)
    elif rank != 4:
        return images

    shape = images.shape
    resize_cfg = ResizeConfig(new_H=size[0], new_W=size[1], H=shape[1], W=shape[2], align_corners=align_corners)
    src_y, src_x = _calculate_nearest_coords(np_mod, resize_cfg)

    out = images[:, src_y[:, None], src_x[None, :], :]

    return out[0] if rank == 3 else out


@numpy_eager_registry.register("ResizeBicubic")
def resize_bicubic(np_mod: object, images: object, size: tuple[int, int], align_corners: bool = False) -> object:
    """Resize images using bicubic interpolation.

    Args:
        np_mod: Numpy module.
        images: The images to resize.
        size: The target size (height, width).
        align_corners: Whether to align corners.

    Returns:
        Any: The resized images.
    """
    return resize_bilinear(np_mod, images, size, align_corners)


@numpy_eager_registry.register("ResizeLanczos3")
def resize_lanczos3(np_mod: object, images: object, size: tuple[int, int], align_corners: bool = False) -> object:
    """Resize images using Lanczos3 interpolation.

    Args:
        np_mod: Numpy module.
        images: The images to resize.
        size: The target size (height, width).
        align_corners: Whether to align corners.

    Returns:
        Any: The resized images.
    """
    return resize_bilinear(np_mod, images, size, align_corners)


@numpy_eager_registry.register("RandomFlip")
def random_flip_numpy(np_mod: object, images: object, mode: str = "horizontal_and_vertical", seed: int = None, **kwargs: object) -> object:
    """Random flip.

    Args:
        np_mod: Numpy module.
        images: Images.
        mode: Mode.
        seed: Seed.
        **kwargs: Kwargs.

    Returns:
        Any: Output.
    """
    np = np_mod
    x = np.asarray(images)
    rank = len(x.shape)
    if rank == 3:
        x = np.expand_dims(x, 0)
    elif rank != 4:
        return x

    rng = np.random.default_rng(seed)
    out = np.copy(x)

    if mode in ("horizontal", "horizontal_and_vertical"):
        mask_h = rng.random(size=(x.shape[0],)) > 0.5
        out[mask_h] = out[mask_h, :, ::-1, :]

    if mode in ("vertical", "horizontal_and_vertical"):
        mask_v = rng.random(size=(x.shape[0],)) > 0.5
        out[mask_v] = out[mask_v, ::-1, :, :]

    if rank == 3:
        return out[0]
    return out


def _compute_rotation_coords(np: object, H: int, W: int, angle: float) -> tuple:
    """Compute coordinates for rotation.

    Args:
        np: Numpy module.
        H: Height.
        W: Width.
        angle: Angle.

    Returns:
        tuple: src_x, src_y.
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

    img: object
    src_x: object
    src_y: object
    fill_mode: str
    fill_value: float


def _get_rotation_coords(np: object, src_x_clip: object, src_y_clip: object, W: int, H: int) -> tuple:
    x0 = np.floor(src_x_clip).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(src_y_clip).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, H - 1)
    return x0, x1, y0, y1


def _gather_rotation_pixels(img: object, coords: tuple) -> tuple:
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


def _compute_rotation_weights(dx: object, dy: object) -> tuple:
    """Compute weights for rotation interpolation.

    Args:
        dx: X delta.
        dy: Y delta.

    Returns:
        tuple: Weights wa, wb, wc, wd.
    """
    wa = (1 - dy)[..., None] * (1 - dx)[..., None]
    wb = dy[..., None] * (1 - dx)[..., None]
    wc = (1 - dy)[..., None] * dx[..., None]
    wd = dy[..., None] * dx[..., None]
    return wa, wb, wc, wd


def _interpolate_rotation(np: object, cfg: RotationInterpolationConfig) -> object:
    """Interpolate rotated image.

    Args:
        np: Numpy module.
        cfg: Configuration.

    Returns:
        Any: Interpolated image.
    """
    H, W, _ = cfg.img.shape
    src_x_clip = np.clip(cfg.src_x, 0, W - 1)
    src_y_clip = np.clip(cfg.src_y, 0, H - 1)

    coords = _get_rotation_coords(np, src_x_clip, src_y_clip, W, H)
    dx = src_x_clip - coords[0]
    dy = src_y_clip - coords[2]

    pixels = _gather_rotation_pixels(cfg.img, coords)
    weights = _compute_rotation_weights(dx, dy)

    interp = pixels[0] * weights[0] + pixels[1] * weights[1] + pixels[2] * weights[2] + pixels[3] * weights[3]
    valid_mask = (cfg.src_x >= 0) & (cfg.src_x <= W - 1) & (cfg.src_y >= 0) & (cfg.src_y <= H - 1)
    if cfg.fill_mode == "constant":
        interp[~valid_mask] = cfg.fill_value
    return interp.astype(cfg.img.dtype)


def _apply_random_rotation_single(np: object, img: object, angle: float, fill_mode: str, fill_value: float) -> object:
    """Apply random rotation to a single image.

    Args:
        np: Numpy module.
        img: Image.
        angle: Angle.
        fill_mode: Fill mode.
        fill_value: Fill value.

    Returns:
        Any: Rotated image.
    """
    H, W, _ = img.shape
    src_x, src_y = _compute_rotation_coords(np, H, W, angle)
    cfg = RotationInterpolationConfig(img=img, src_x=src_x, src_y=src_y, fill_mode=fill_mode, fill_value=fill_value)
    return _interpolate_rotation(np, cfg)


@dataclass
class TransformOptions:
    """Options for image transformations."""

    fill_mode: str = "reflect"
    interpolation: str = "bilinear"
    seed: Optional[int] = None
    fill_value: float = 0.0


def _apply_rotation_to_batch(np: object, x: object, rng: object, angles: tuple[float, float], options: TransformOptions) -> object:
    """Apply rotation to batch of images.

    Args:
        np: Numpy module.
        x: Images batch.
        rng: Random number generator.
        angles: Tuple of lower, upper angle.
        options: Transform options.

    Returns:
        Any: Rotated batch.
    """
    B = x.shape[0]
    out = np.zeros_like(x)
    for i in range(B):
        angle = rng.uniform(angles[0], angles[1]) * 2 * np.pi
        out[i] = _apply_random_rotation_single(np, x[i], angle, options.fill_mode, options.fill_value)
    return out


@numpy_eager_registry.register("RandomRotation")
def random_rotation_numpy(np_mod: object, images: object, factor: float, options: Optional[TransformOptions] = None, **kwargs: object) -> object:
    """Random rotation.

    Args:
        np_mod: Numpy module.
        images: Images.
        factor: Factor.
        options: Transformation options.
        **kwargs: Kwargs.

    Returns:
        Any: Output.
    """
    options = options or TransformOptions()
    np = np_mod
    x = np.asarray(images)
    rank = len(x.shape)
    if rank == 3:
        x = np.expand_dims(x, 0)
    elif rank != 4:
        return x

    rng = np.random.default_rng(options.seed)

    if isinstance(factor, (tuple, list)):
        lower, upper = factor
    else:
        lower, upper = -factor, factor

    out = _apply_rotation_to_batch(np, x, rng, (lower, upper), options)

    if rank == 3:
        return out[0]
    return out
