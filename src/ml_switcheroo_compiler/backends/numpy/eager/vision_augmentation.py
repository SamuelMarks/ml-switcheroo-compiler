"""Shared vision utilities and ops."""

from __future__ import annotations

from collections.abc import Callable

import pydantic

from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    random_crop_eager,
    random_translation_eager,
    random_zoom_eager,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    ensure_callable,
    ensure_eager_tensor,
    ensure_float,
    ensure_int,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, ensure_mod, get_backend_member
from ml_switcheroo_compiler.core.shape import TensorOrScalar as TensorOrScalar


@numpy_eager_registry.register("AugMix")
def _np_augmix(backend_module: BackendModuleProtocol, images: EagerValue, factor: float, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np augmix operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images (EagerValue): Input `images` argument (images).
        factor (float): Input `factor` argument (factor).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images


@numpy_eager_registry.register("Cutmix")
def _np_cutmix(backend_module: BackendModuleProtocol, images1: EagerValue, images2: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cutmix operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images1 (EagerValue): Input `images1` argument (images1).
        images2 (EagerValue): Input `images2` argument (images2).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images1


@numpy_eager_registry.register("Mixup")
def _np_mixup(backend_module: BackendModuleProtocol, images1: EagerValue, images2: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np mixup operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images1 (EagerValue): Input `images1` argument (images1).
        images2 (EagerValue): Input `images2` argument (images2).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images1


@numpy_eager_registry.register("RandAugment")
def _np_rand_augment(backend_module: BackendModuleProtocol, images: EagerValue, factor: float, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rand augment operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images (EagerValue): Input `images` argument (images).
        factor (float): Input `factor` argument (factor).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images


@numpy_eager_registry.register("RandomColorJitter")
def _np_random_color_jitter(backend_module: BackendModuleProtocol, images: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np random color jitter operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images (EagerValue): Input `images` argument (images).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images


@numpy_eager_registry.register("RandomCrop")
def _np_random_crop(backend_module: BackendModuleProtocol, images: EagerValue, size: EagerValue, seed: EagerValue = None) -> EagerValue:
    """Execute the eager np random crop operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images (EagerValue): Input `images` argument (images).
        size (tuple): Input `size` argument (size).
        seed (EagerValue): Input `seed` argument (seed).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(random_crop_eager)(backend_module, images, size, seed)


@numpy_eager_registry.register("RandomErasing")
def _np_random_erasing(backend_module: BackendModuleProtocol, images: EagerValue, factor: float, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np random erasing operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images (EagerValue): Input `images` argument (images).
        factor (float): Input `factor` argument (factor).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images


# @numpy_eager_registry.register("RandomFlip")
def _np_flip_horizontal(img: EagerValue, rng: EagerValue) -> EagerValue:
    """Flip an image horizontally with 50% probability.

    Args:
        img: The image array to flip.
        rng: Random number generator.

    Returns:
        The flipped or unchanged image array.
    """
    return ensure_eager_tensor(img)[:, :, ::-1] if ensure_mod(rng).random() > 0.5 else img


def _np_flip_vertical(img: EagerValue, rng: EagerValue) -> EagerValue:
    """Flip an image vertically with 50% probability.

    Args:
        img: The image array to flip.
        rng: Random number generator.

    Returns:
        The flipped or unchanged image array.
    """
    return ensure_eager_tensor(img)[:, ::-1, :] if ensure_mod(rng).random() > 0.5 else img


def _np_flip_both(img: EagerValue, rng: EagerValue) -> EagerValue:
    """Flip an image both horizontally and vertically with 50% probability each.

    Args:
        img: The image array to flip.
        rng: Random number generator.

    Returns:
        The flipped or unchanged image array.
    """
    img = _np_flip_horizontal(img, rng)
    return _np_flip_vertical(img, rng)


_NP_FLIP_STRATEGIES: dict[str, Callable[[EagerValue, EagerValue], EagerValue]] = {
    "horizontal": _np_flip_horizontal,
    "vertical": _np_flip_vertical,
    "horizontal_and_vertical": _np_flip_both,
}


def _np_random_flip(images: EagerValue, mode: str, seed: EagerValue = None) -> EagerValue:
    """Execute the eager np random flip operation on the active backend array.

    Args:
        images (EagerValue): Input `images` argument (images).
        mode (str): Input `mode` argument (mode).
        seed (EagerValue): Input `seed` argument (seed).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import random

    import numpy as np

    rng = get_backend_member(random, "Random")(seed)

    # Check if a single image
    strategy: Callable[[EagerValue, EagerValue], EagerValue] = _NP_FLIP_STRATEGIES.get(mode, lambda img, r: img)
    if len(get_backend_member(images, "shape")) == 3:
        return strategy(images, rng)

    out = get_backend_member(np, "copy")(images)

    for i in range(out.shape[0]):
        out[i] = strategy(out[i], rng)

    return ensure_eager_tensor(out)


class _PositionalVisionModel(pydantic.BaseModel):
    """Base vision model supporting positional and keyword initialization."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Initialize model from positional or keyword arguments.

        Args:
            *args (EagerValue): Positional field values.
            **kwargs (EagerValue): Keyword field values.
        """
        pos_dict: dict[str, EagerValue] = {k: args[idx] for idx, k in enumerate(type(self).model_fields.keys()) if idx < len(args)}
        super().__init__(**{**pos_dict, **kwargs})


class RotationConfig(_PositionalVisionModel):
    """Rotation configuration."""

    theta: float
    H: int
    W: int
    x: EagerValue
    y: EagerValue


def _calculate_rotation_matrix(np: BackendModuleProtocol, cfg: RotationConfig) -> EagerValue:
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


def _nearest_interpolation(
    np: EagerValue,
    images: EagerValue,
    coords: tuple[EagerValue, EagerValue],
    shape: tuple[int, int],
    b_c: tuple[int, int],
) -> EagerValue:
    """Nearest neighbor interpolation.

    Args:
        np (EagerValue): Input `np` argument (np).
        images (EagerValue): Input `images` argument (images).
        coords (tuple): Input `coords` argument (coords).
        shape (tuple): Input `shape` argument (shape).
        b_c (tuple): Input `b_c` argument (b c).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    src_x, src_y = coords
    H, W = shape
    b, c = b_c
    mod = ensure_mod(np)
    src_y_round = mod.clip(mod.round(src_y).astype(mod.int32), 0, H - 1)
    src_x_round = mod.clip(mod.round(src_x).astype(mod.int32), 0, W - 1)
    return ensure_eager_tensor(images)[b, src_y_round, src_x_round, c]


class InterpPixelsConfig(_PositionalVisionModel):
    """InterpPixelsConfig class."""

    images: EagerValue
    y0: EagerValue
    y1: EagerValue
    x0: EagerValue
    x1: EagerValue
    H: int
    W: int
    b_c: tuple[EagerValue, ...]


def _get_interp_pixels(np: BackendModuleProtocol, cfg: InterpPixelsConfig) -> EagerValue:
    """Gather pixels.

    Args:
        np (ModuleType): Input `np` argument (np).
        cfg (InterpPixelsConfig): Input `cfg` argument (cfg).

    Returns:
        tuple: Computed `tuple` output.
    """
    b, c = cfg.b_c
    y0_c = np.clip(cfg.y0, 0, cfg.H - 1)
    y1_c = np.clip(cfg.y1, 0, cfg.H - 1)
    x0_c = np.clip(cfg.x0, 0, cfg.W - 1)
    x1_c = np.clip(cfg.x1, 0, cfg.W - 1)
    Ia = ensure_eager_tensor(cfg.images)[b, y0_c, x0_c, c]
    Ib = ensure_eager_tensor(cfg.images)[b, y1_c, x0_c, c]
    Ic = ensure_eager_tensor(cfg.images)[b, y0_c, x1_c, c]
    Id = ensure_eager_tensor(cfg.images)[b, y1_c, x1_c, c]
    return Ia, Ib, Ic, Id


def _get_interp_weights(
    src_coords: tuple[EagerValue, EagerValue],
    bounds: tuple[EagerValue, EagerValue, EagerValue, EagerValue],
) -> tuple[EagerValue, EagerValue, EagerValue, EagerValue]:
    """Execute the eager get interp weights operation on the active backend array.

    Args:
        src_coords (tuple): Input `src_coords` argument (src coords).
        bounds (tuple): Input `bounds` argument (bounds).

    Returns:
        tuple: Computed `tuple` output.
    """
    src_y = ensure_eager_tensor(src_coords[0])
    src_x = ensure_eager_tensor(src_coords[1])
    y0 = ensure_eager_tensor(bounds[0])
    y1 = ensure_eager_tensor(bounds[1])
    x0 = ensure_eager_tensor(bounds[2])
    x1 = ensure_eager_tensor(bounds[3])
    wa = (y1 - src_y) * (x1 - src_x)
    wb = (src_y - y0) * (x1 - src_x)
    wc = (y1 - src_y) * (src_x - x0)
    wd = (src_y - y0) * (src_x - x0)
    return wa, wb, wc, wd


def _bilinear_interpolation(
    np: EagerValue,
    images: EagerValue,
    coords: tuple[EagerValue, EagerValue],
    shape: tuple[int, int],
    b_c: tuple[int, int],
) -> EagerValue:
    """Bilinear interpolation.

    Args:
        np (EagerValue): Input `np` argument (np).
        images (EagerValue): Input `images` argument (images).
        coords (tuple): Input `coords` argument (coords).
        shape (tuple): Input `shape` argument (shape).
        b_c (tuple): Input `b_c` argument (b c).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    mod = ensure_mod(np)
    y0: EagerTensorProtocol = mod.floor(coords[1]).astype(mod.int32)
    y1: EagerTensorProtocol = y0 + 1
    x0: EagerTensorProtocol = mod.floor(coords[0]).astype(mod.int32)
    x1: EagerTensorProtocol = x0 + 1

    pixel_cfg = InterpPixelsConfig(images=images, y0=y0, y1=y1, x0=x0, x1=x1, H=shape[0], W=shape[1], b_c=b_c)
    pixels = _get_interp_pixels(mod, pixel_cfg)
    weights = _get_interp_weights((coords[1], coords[0]), (y0, y1, x0, x1))

    from ml_switcheroo_compiler.backends.eager_registry import ensure_tensor_quad

    p0, p1, p2, p3 = ensure_tensor_quad(pixels)
    w0, w1, w2, w3 = ensure_tensor_quad(weights)
    return ensure_eager_tensor(p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3)


class AffineConfig(_PositionalVisionModel):
    """Affine configuration."""

    coords: tuple[EagerValue, ...]
    shape: tuple[EagerValue, ...]
    b_c: tuple[EagerValue, ...]
    options: tuple[EagerValue, ...]


def _apply_affine_grid(np: BackendModuleProtocol, images: EagerValue, cfg: AffineConfig) -> EagerValue:
    """Apply affine grid sampling.

    Args:
        np: The numpy module.
        images: Input images.
        cfg: Affine configuration.

    Returns: np.ndarray: Sampled values.
    """
    interpolation, fill_mode, fill_value, mask = cfg.options
    if interpolation == "nearest":
        val = ensure_callable(_nearest_interpolation)(np, images, cfg.coords, cfg.shape, cfg.b_c)
    else:
        val = ensure_callable(_bilinear_interpolation)(np, images, cfg.coords, cfg.shape, cfg.b_c)

    if fill_mode == "constant":
        val = get_backend_member(np, "where")(mask, val, fill_value)

    return ensure_eager_tensor(get_backend_member(val, "astype")(get_backend_member(images, "dtype")))


class BatchRotationConfig(_PositionalVisionModel):
    """Batch rotation configuration."""

    images: EagerTensorProtocol
    angles: EagerTensorProtocol
    H: int
    W: int
    x: EagerValue
    y: EagerValue
    options: tuple[EagerValue, ...]


def _process_batch_item(np: BackendModuleProtocol, cfg: BatchRotationConfig, b: int, out: EagerValue) -> None:
    """Process a single item in the batch.

    Args:
        np (ModuleType): Input `np` argument (np).
        cfg (BatchRotationConfig): Input `cfg` argument (cfg).
        b (int): Input `b` argument (b).
        out (EagerValue): Input `out` argument (out).
    """
    from ml_switcheroo_compiler.backends.eager_registry import ensure_tensor_pair

    rot_cfg = RotationConfig(theta=ensure_float(cfg.angles[b]), H=cfg.H, W=cfg.W, x=cfg.x, y=cfg.y)
    src_x, src_y = ensure_tensor_pair(_calculate_rotation_matrix(np, rot_cfg))
    mask = (src_x >= 0) & (src_x <= cfg.W - 1) & (src_y >= 0) & (src_y <= cfg.H - 1)
    C = int(tuple(cfg.images.shape)[3])
    for c in range(C):
        aff_cfg = AffineConfig(coords=(src_x, src_y), shape=(cfg.H, cfg.W), b_c=(b, c), options=cfg.options + (mask,))
        ensure_eager_tensor(out)[b, :, :, c] = _apply_affine_grid(np, cfg.images, aff_cfg)


def _apply_rotation_batch(np: BackendModuleProtocol, cfg: BatchRotationConfig) -> EagerValue:
    """Apply rotation to a batch of images.

    Args:
        np (ModuleType): Input `np` argument (np).
        cfg (BatchRotationConfig): Input `cfg` argument (cfg).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    B = int(tuple(cfg.images.shape)[0])
    out = np.zeros_like(cfg.images, dtype=cfg.images.dtype)
    for b in range(B):
        _process_batch_item(np, cfg, b, out)
    return ensure_eager_tensor(out)


def _resolve_rotation_factor(
    factor: float | tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Resolve rotation bounds.

    Args:
        factor (float | tuple[float, float] | list[float]): Input `factor` argument (factor).

    Returns:
        tuple: Computed `tuple` output.
    """
    if isinstance(factor, (list, tuple)):
        return factor[0], factor[1]
    return -factor, factor


def _create_rotation_mesh(np: BackendModuleProtocol, H: int, W: int) -> EagerValue:
    """Create mesh grid for rotation.

    Args:
        np (ModuleType): Input `np` argument (np).
        H (int): Input `H` argument (H).
        W (int): Input `W` argument (W).

    Returns:
        tuple: Computed `tuple` output.
    """
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y = y.astype(np.float32) - H / 2.0
    x = x.astype(np.float32) - W / 2.0
    return y, x


@numpy_eager_registry.register("RandomRotation")
def _np_random_rotation(backend_module: BackendModuleProtocol, images: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np random rotation operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images (EagerValue): Input `images` argument (images).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    np = backend_module
    images = np.asarray(images)
    rank = len(get_backend_member(images, "shape"))
    if rank == 3:
        images = np.expand_dims(images, 0)
    elif rank != 4:
        return images

    raw_b, raw_h, raw_w, _ = get_backend_member(images, "shape")
    B, H, W = ensure_int(raw_b), ensure_int(raw_h), ensure_int(raw_w)

    from ml_switcheroo_compiler.backends.eager_registry import ensure_tensor_pair

    rng = np.random.default_rng(kwargs.get("seed", None))
    angles = rng.uniform(*ensure_callable(_resolve_rotation_factor)(kwargs.get("factor", 0.0)), size=(B,)) * 2 * np.pi

    y, x = ensure_tensor_pair(_create_rotation_mesh(np, H, W))

    options = (
        kwargs.get("interpolation", "bilinear"),
        kwargs.get("fill_mode", "reflect"),
        kwargs.get("fill_value", 0.0),
    )

    batch_cfg = BatchRotationConfig(images=ensure_eager_tensor(images), angles=angles, H=H, W=W, x=x, y=y, options=options)
    out = _apply_rotation_batch(np, batch_cfg)

    return ensure_eager_tensor(out)[0] if rank == 3 else out


@numpy_eager_registry.register("RandomTranslation")
def _np_random_translation(backend_module: BackendModuleProtocol, images: EagerValue, height_factor: EagerValue, width_factor: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np random translation operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images (EagerValue): Input `images` argument (images).
        height_factor (EagerValue): Input `height_factor` argument (height factor).
        width_factor (EagerValue): Input `width_factor` argument (width factor).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(random_translation_eager)(backend_module, images, height_factor, width_factor, **kwargs)


@numpy_eager_registry.register("RandomZoom")
def _np_random_zoom(backend_module: BackendModuleProtocol, images: EagerValue, height_factor: EagerValue, width_factor: EagerValue = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np random zoom operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        images (EagerValue): Input `images` argument (images).
        height_factor (EagerValue): Input `height_factor` argument (height factor).
        width_factor (EagerValue): Input `width_factor` argument (width factor).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(random_zoom_eager)(backend_module, images, height_factor, width_factor, **kwargs)


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
