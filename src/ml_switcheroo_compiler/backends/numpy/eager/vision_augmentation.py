# ruff: noqa: E501
"""Shared vision utilities and ops."""

from dataclasses import dataclass

from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    random_crop_eager,
    random_translation_eager,
    random_zoom_eager,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AugMix")
def _np_augmix(backend_module: object, images: object, factor: float, **kwargs: object) -> object:
    """Evaluate _np_augmix operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        factor (float): The factor parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return images


@numpy_eager_registry.register("Cutmix")
def _np_cutmix(backend_module: object, images1: object, images2: object, **kwargs: object) -> object:
    """Evaluate _np_cutmix operation.

    Args:
        backend_module (object): The backend_module parameter.
        images1 (object): The images1 parameter.
        images2 (object): The images2 parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return images1


@numpy_eager_registry.register("Mixup")
def _np_mixup(backend_module: object, images1: object, images2: object, **kwargs: object) -> object:
    """Evaluate _np_mixup operation.

    Args:
        backend_module (object): The backend_module parameter.
        images1 (object): The images1 parameter.
        images2 (object): The images2 parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return images1


@numpy_eager_registry.register("RandAugment")
def _np_rand_augment(backend_module: object, images: object, factor: float, **kwargs: object) -> object:
    """Evaluate _np_rand_augment operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        factor (float): The factor parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return images


@numpy_eager_registry.register("RandomColorJitter")
def _np_random_color_jitter(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_random_color_jitter operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return images


@numpy_eager_registry.register("RandomCrop")
def _np_random_crop(backend_module: object, images: object, size: tuple, seed: object = None) -> object:
    """Evaluate _np_random_crop operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        size (tuple): The size parameter.
        seed (object): The seed parameter.

    Returns:
        object: Result.
    """
    return random_crop_eager(backend_module, images, size, seed)


@numpy_eager_registry.register("RandomErasing")
def _np_random_erasing(backend_module: object, images: object, factor: float, **kwargs: object) -> object:
    """Evaluate _np_random_erasing operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        factor (float): The factor parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return images


# @numpy_eager_registry.register("RandomFlip")
def _np_flip_horizontal(img: object, rng: object) -> object:
    """Flip an image horizontally with 50% probability.

    Args:
        img: The image array to flip.
        rng: Random number generator.

    Returns:
        The flipped or unchanged image array.
    """
    return img[:, :, ::-1] if rng.random() > 0.5 else img


def _np_flip_vertical(img: object, rng: object) -> object:
    """Flip an image vertically with 50% probability.

    Args:
        img: The image array to flip.
        rng: Random number generator.

    Returns:
        The flipped or unchanged image array.
    """
    return img[:, ::-1, :] if rng.random() > 0.5 else img


def _np_flip_both(img: object, rng: object) -> object:
    """Flip an image both horizontally and vertically with 50% probability each.

    Args:
        img: The image array to flip.
        rng: Random number generator.

    Returns:
        The flipped or unchanged image array.
    """
    img = _np_flip_horizontal(img, rng)
    return _np_flip_vertical(img, rng)


_NP_FLIP_STRATEGIES = {
    "horizontal": _np_flip_horizontal,
    "vertical": _np_flip_vertical,
    "horizontal_and_vertical": _np_flip_both,
}


def _np_random_flip(images: object, mode: str, seed: object = None) -> object:
    """Evaluate _np_random_flip operation.

    Args:
        images (object): The images parameter.
        mode (str): The mode parameter.
        seed (object): The seed parameter.

    Returns:
        object: Result.
    """
    import random

    import numpy as np

    rng = random.Random(seed)

    # Check if a single image
    if len(images.shape) == 3:
        strategy = _NP_FLIP_STRATEGIES.get(mode, lambda img, r: img)
        return strategy(images, rng)

    out = np.copy(images)
    strategy = _NP_FLIP_STRATEGIES.get(mode, lambda img, r: img)

    for i in range(out.shape[0]):
        out[i] = strategy(out[i], rng)

    return out


@dataclass
class RotationConfig:
    """Rotation configuration."""

    theta: float
    H: int
    W: int
    x: object
    y: object


def _calculate_rotation_matrix(np: object, cfg: RotationConfig) -> tuple:
    """Calculate rotation matrix coordinates.

    Args:
        np: The numpy module.
        cfg: Rotation configuration.

    Returns:
        tuple: src_x, src_y.
    """
    cos_t = np.cos(cfg.theta)
    sin_t = np.sin(cfg.theta)

    src_x = cos_t * cfg.x - sin_t * cfg.y + cfg.W / 2.0
    src_y = sin_t * cfg.x + cos_t * cfg.y + cfg.H / 2.0
    return src_x, src_y


def _nearest_interpolation(np: object, images: object, coords: tuple, shape: tuple, b_c: tuple) -> object:
    """Nearest neighbor interpolation.

    Args:
        np (object): The np parameter.
        images (object): The images parameter.
        coords (tuple): The coords parameter.
        shape (tuple): The shape parameter.
        b_c (tuple): The b_c parameter.

    Returns:
        object: Result.
    """
    src_x, src_y = coords
    H, W = shape
    b, c = b_c
    src_y_round = np.clip(np.round(src_y).astype(np.int32), 0, H - 1)
    src_x_round = np.clip(np.round(src_x).astype(np.int32), 0, W - 1)
    return images[b, src_y_round, src_x_round, c]


@dataclass
class InterpPixelsConfig:
    """InterpPixelsConfig class."""

    images: object
    y0: object
    y1: object
    x0: object
    x1: object
    H: int
    W: int
    b_c: tuple


def _get_interp_pixels(np: object, cfg: InterpPixelsConfig) -> tuple:
    """Gather pixels.

    Args:
        np (object): The np parameter.
        cfg (InterpPixelsConfig): The cfg parameter.

    Returns:
        tuple: Result.
    """
    b, c = cfg.b_c
    y0_c = np.clip(cfg.y0, 0, cfg.H - 1)
    y1_c = np.clip(cfg.y1, 0, cfg.H - 1)
    x0_c = np.clip(cfg.x0, 0, cfg.W - 1)
    x1_c = np.clip(cfg.x1, 0, cfg.W - 1)
    Ia = cfg.images[b, y0_c, x0_c, c]
    Ib = cfg.images[b, y1_c, x0_c, c]
    Ic = cfg.images[b, y0_c, x1_c, c]
    Id = cfg.images[b, y1_c, x1_c, c]
    return Ia, Ib, Ic, Id


def _get_interp_weights(src_coords: tuple, bounds: tuple) -> tuple:
    """Evaluate _get_interp_weights operation.

    Args:
        src_coords (tuple): The src_coords parameter.
        bounds (tuple): The bounds parameter.

    Returns:
        tuple: Result.
    """
    src_y, src_x = src_coords
    y0, y1, x0, x1 = bounds
    wa = (y1 - src_y) * (x1 - src_x)
    wb = (src_y - y0) * (x1 - src_x)
    wc = (y1 - src_y) * (src_x - x0)
    wd = (src_y - y0) * (src_x - x0)
    return wa, wb, wc, wd


def _bilinear_interpolation(np: object, images: object, coords: tuple, shape: tuple, b_c: tuple) -> object:
    """Bilinear interpolation.

    Args:
        np (object): The np parameter.
        images (object): The images parameter.
        coords (tuple): The coords parameter.
        shape (tuple): The shape parameter.
        b_c (tuple): The b_c parameter.

    Returns:
        object: Result.
    """
    y0 = np.floor(coords[1]).astype(np.int32)
    y1 = y0 + 1
    x0 = np.floor(coords[0]).astype(np.int32)
    x1 = x0 + 1

    pixel_cfg = InterpPixelsConfig(images=images, y0=y0, y1=y1, x0=x0, x1=x1, H=shape[0], W=shape[1], b_c=b_c)
    pixels = _get_interp_pixels(np, pixel_cfg)
    weights = _get_interp_weights((coords[1], coords[0]), (y0, y1, x0, x1))

    return pixels[0] * weights[0] + pixels[1] * weights[1] + pixels[2] * weights[2] + pixels[3] * weights[3]


@dataclass
class AffineConfig:
    """Affine configuration."""

    coords: tuple
    shape: tuple
    b_c: tuple
    options: tuple


def _apply_affine_grid(np: object, images: object, cfg: AffineConfig) -> object:
    """Apply affine grid sampling.

    Args:
        np: The numpy module.
        images: Input images.
        cfg: Affine configuration.

    Returns:
        object: Sampled values.
    """
    interpolation, fill_mode, fill_value, mask = cfg.options
    if interpolation == "nearest":
        val = _nearest_interpolation(np, images, cfg.coords, cfg.shape, cfg.b_c)
    else:
        val = _bilinear_interpolation(np, images, cfg.coords, cfg.shape, cfg.b_c)

    if fill_mode == "constant":
        val = np.where(mask, val, fill_value)

    return val.astype(images.dtype)


@dataclass
class BatchRotationConfig:
    """Batch rotation configuration."""

    images: object
    angles: object
    H: int
    W: int
    x: object
    y: object
    options: tuple


def _process_batch_item(np: object, cfg: BatchRotationConfig, b: int, out: object) -> None:
    """Process a single item in the batch.

    Args:
        np (object): The np parameter.
        cfg (BatchRotationConfig): The cfg parameter.
        b (int): The b parameter.
        out (object): The out parameter.
    """
    rot_cfg = RotationConfig(theta=cfg.angles[b], H=cfg.H, W=cfg.W, x=cfg.x, y=cfg.y)
    src_x, src_y = _calculate_rotation_matrix(np, rot_cfg)
    mask = (src_x >= 0) & (src_x <= cfg.W - 1) & (src_y >= 0) & (src_y <= cfg.H - 1)
    C = cfg.images.shape[3]
    for c in range(C):
        aff_cfg = AffineConfig(coords=(src_x, src_y), shape=(cfg.H, cfg.W), b_c=(b, c), options=cfg.options + (mask,))
        out[b, :, :, c] = _apply_affine_grid(np, cfg.images, aff_cfg)


def _apply_rotation_batch(np: object, cfg: BatchRotationConfig) -> object:
    """Apply rotation to a batch of images.

    Args:
        np (object): The np parameter.
        cfg (BatchRotationConfig): The cfg parameter.

    Returns:
        object: Result.
    """
    B = cfg.images.shape[0]
    out = np.zeros_like(cfg.images, dtype=cfg.images.dtype)
    for b in range(B):
        _process_batch_item(np, cfg, b, out)
    return out


def _resolve_rotation_factor(factor: object) -> tuple:
    """Resolve rotation bounds.

    Args:
        factor (object): The factor parameter.

    Returns:
        tuple: Result.
    """
    if isinstance(factor, (list, tuple)):
        return factor[0], factor[1]
    return -factor, factor


def _create_rotation_mesh(np: object, H: int, W: int) -> tuple:
    """Create mesh grid for rotation.

    Args:
        np (object): The np parameter.
        H (int): The H parameter.
        W (int): The W parameter.

    Returns:
        tuple: Result.
    """
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y = y.astype(np.float32) - H / 2.0
    x = x.astype(np.float32) - W / 2.0
    return y, x


@numpy_eager_registry.register("RandomRotation")
def _np_random_rotation(backend_module: object, images: object, **kwargs: object) -> object:
    """Evaluate _np_random_rotation operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    np = backend_module
    images = np.asarray(images)
    rank = len(images.shape)
    if rank == 3:
        images = np.expand_dims(images, 0)
    elif rank != 4:
        return images

    B, H, W, _ = images.shape

    rng = np.random.default_rng(kwargs.get("seed", None))
    angles = rng.uniform(*_resolve_rotation_factor(kwargs.get("factor", 0.0)), size=(B,)) * 2 * np.pi

    y, x = _create_rotation_mesh(np, H, W)

    options = (
        kwargs.get("interpolation", "bilinear"),
        kwargs.get("fill_mode", "reflect"),
        kwargs.get("fill_value", 0.0),
    )

    batch_cfg = BatchRotationConfig(images=images, angles=angles, H=H, W=W, x=x, y=y, options=options)
    out = _apply_rotation_batch(np, batch_cfg)

    return out[0] if rank == 3 else out


@numpy_eager_registry.register("RandomTranslation")
def _np_random_translation(backend_module: object, images: object, height_factor: object, width_factor: object, **kwargs: object) -> object:
    """Evaluate _np_random_translation operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        height_factor (object): The height_factor parameter.
        width_factor (object): The width_factor parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return random_translation_eager(backend_module, images, height_factor, width_factor, **kwargs)


@numpy_eager_registry.register("RandomZoom")
def _np_random_zoom(backend_module: object, images: object, height_factor: object, width_factor: object = None, **kwargs: object) -> object:
    """Evaluate _np_random_zoom operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        height_factor (object): The height_factor parameter.
        width_factor (object): The width_factor parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    return random_zoom_eager(backend_module, images, height_factor, width_factor, **kwargs)


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_np_augmix",
    "_np_cutmix",
    "_np_mixup",
    "_np_rand_augment",
    "_np_random_color_jitter",
    "_np_random_crop",
    "_np_random_erasing",
    "_np_random_flip",
    "_np_random_rotation",
    "_np_random_translation",
    "_np_random_zoom",
    "numpy_eager_registry",
]
