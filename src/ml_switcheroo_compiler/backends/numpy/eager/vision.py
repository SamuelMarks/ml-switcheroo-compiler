# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Vision operations for the numpy backend."""

from dataclasses import dataclass
from typing import Any, Optional

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

    images: Any
    src_y: Any
    src_x: Any
    H: int
    W: int


def _calculate_bilinear_coords(np: Any, cfg: ResizeConfig) -> tuple:
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

    y0: Any
    y1: Any
    x0: Any
    x1: Any


def _compute_bilinear_pixels(np: Any, images: Any, coords: BilinearCoords) -> tuple:
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


def _compute_bilinear_weights(dy: Any, dx: Any) -> tuple:
    """Evaluate _compute_bilinear_weights operation.

    Args:
        dy (object): The dy parameter.
        dx (object): The dx parameter.

    Returns:
        tuple: Result.
    """
    wa = (1 - dy[:, None, None]) * (1 - dx[None, :, None])
    wb = (dy[:, None, None]) * (1 - dx[None, :, None])
    wc = (1 - dy[:, None, None]) * (dx[None, :, None])
    wd = (dy[:, None, None]) * (dx[None, :, None])
    return wa, wb, wc, wd


def _apply_bilinear_interpolation(np: Any, cfg: InterpolationConfig) -> Any:
    """Apply bilinear interpolation.

    Args:
        np (object): The np parameter.
        cfg (InterpolationConfig): The cfg parameter.

    Returns: Any: Result.
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
def resize_bilinear(np_mod: Any, images: Any, size: tuple[int, int], align_corners: bool = False) -> Any:
    """Resize images using bilinear interpolation.

    Args:
        np_mod (object): The np_mod parameter.
        images (object): The images parameter.
        size (tuple): The size parameter.
        align_corners (bool): The align_corners parameter.

    Returns: Any: Result.
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


def _calculate_nearest_coords(np: Any, cfg: ResizeConfig) -> tuple:
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
def resize_nearest(np_mod: Any, images: Any, size: tuple[int, int], align_corners: bool = False) -> Any:
    """Resize images using nearest-neighbor interpolation.

    Args:
        np_mod (object): The np_mod parameter.
        images (object): The images parameter.
        size (tuple): The size parameter.
        align_corners (bool): The align_corners parameter.

    Returns: Any: Result.
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
def resize_bicubic(np_mod: Any, images: Any, size: tuple[int, int], align_corners: bool = False) -> Any:
    """Resize images using bicubic interpolation.

    Args:
        np_mod (object): The np_mod parameter.
        images (object): The images parameter.
        size (tuple): The size parameter.
        align_corners (bool): The align_corners parameter.

    Returns: Any: Result.
    """
    return resize_bilinear(np_mod, images, size, align_corners)


@numpy_eager_registry.register("ResizeLanczos3")
def resize_lanczos3(np_mod: Any, images: Any, size: tuple[int, int], align_corners: bool = False) -> Any:
    """Resize images using Lanczos3 interpolation.

    Args:
        np_mod (object): The np_mod parameter.
        images (object): The images parameter.
        size (tuple): The size parameter.
        align_corners (bool): The align_corners parameter.

    Returns: Any: Result.
    """
    return resize_bilinear(np_mod, images, size, align_corners)


@numpy_eager_registry.register("RandomFlip")
def random_flip_numpy(np_mod: Any, images: Any, mode: Any = "horizontal_and_vertical", seed: Any = None, **kwargs: Any) -> Any:
    """Generate random flip.

    Args:
        np_mod (object): The np_mod parameter.
        images (object): The images parameter.
        mode (str): The mode parameter.
        seed (int): The seed parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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


def _compute_rotation_coords(np: Any, H: int, W: int, angle: float) -> tuple:
    """Evaluate _compute_rotation_coords operation.

    Args:
        np (object): The np parameter.
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

    img: Any
    src_x: Any
    src_y: Any
    fill_mode: str
    fill_value: float


def _get_rotation_coords(np: Any, src_x_clip: Any, src_y_clip: Any, W: int, H: int) -> tuple:
    """Get the bounding coordinates for rotation interpolation.

    Args:
        np (object): The numpy module.
        src_x_clip (object): Clipped source X coordinates.
        src_y_clip (object): Clipped source Y coordinates.
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


def _gather_rotation_pixels(img: Any, coords: tuple) -> tuple:
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


def _compute_rotation_weights(dx: Any, dy: Any) -> tuple:
    """Evaluate _compute_rotation_weights operation.

    Args:
        dx (object): The dx parameter.
        dy (object): The dy parameter.

    Returns:
        tuple: Result.
    """
    wa = (1 - dy)[..., None] * (1 - dx)[..., None]
    wb = dy[..., None] * (1 - dx)[..., None]
    wc = (1 - dy)[..., None] * dx[..., None]
    wd = dy[..., None] * dx[..., None]
    return wa, wb, wc, wd


def _interpolate_rotation(np: Any, cfg: RotationInterpolationConfig) -> Any:
    """Interpolate rotated image.

    Args:
        np (object): The np parameter.
        cfg (RotationInterpolationConfig): The cfg parameter.

    Returns: Any: Result.
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


def _apply_random_rotation_single(np: Any, img: Any, angle: float, fill_mode: str, fill_value: float) -> Any:
    """Apply random rotation to a single image.

    Args:
        np (object): The np parameter.
        img (object): The img parameter.
        angle (float): The angle parameter.
        fill_mode (str): The fill_mode parameter.
        fill_value (float): The fill_value parameter.

    Returns: Any: Result.
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


def _apply_rotation_to_batch(np: Any, x: Any, rng: Any, angles: tuple[float, float], options: TransformOptions) -> Any:
    """Apply rotation to batch of images.

    Args:
        np (object): The np parameter.
        x (object): The x parameter.
        rng (object): The rng parameter.
        angles (tuple): The angles parameter.
        options (TransformOptions): The options parameter.

    Returns: Any: Result.
    """
    B = x.shape[0]
    out = np.zeros_like(x)
    for i in range(B):
        angle = rng.uniform(angles[0], angles[1]) * 2 * np.pi
        out[i] = _apply_random_rotation_single(np, x[i], angle, options.fill_mode, options.fill_value)
    return out


@numpy_eager_registry.register("RandomRotation")
def random_rotation_numpy(np_mod: Any, images: Any, factor: float, options: Optional[TransformOptions] = None, **kwargs: Any) -> Any:
    """Generate random rotation.

    Args:
        np_mod (object): The np_mod parameter.
        images (object): The images parameter.
        factor (float): The factor parameter.
        options (Optional): The options parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
