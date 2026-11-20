"""Generic backend-parametric eager reference operations for `vision_ops.augmentation`."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable

import pydantic

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    random_crop_eager,
    random_translation_eager,
    random_zoom_eager,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendArrayProtocol,
    EagerTensorProtocol,
    coerce_array,
    coerce_float,
    coerce_int,
    coerce_numeric,
    coerce_shape,
    global_eager_registry,
    unpack_pair,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.shape import OpValue as OpValue


@global_eager_registry.register("AugMix")
def _np_augmix(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    factor: float,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np augmix operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images (BackendArg): Input `images` argument (images).
        factor (float): Input `factor` argument (factor).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images


@global_eager_registry.register("Cutmix")
def _np_cutmix(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images1: BackendArg,
    images2: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np cutmix operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images1 (BackendArg): Input `images1` argument (images1).
        images2 (BackendArg): Input `images2` argument (images2).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images1


@global_eager_registry.register("Mixup")
def _np_mixup(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images1: BackendArg,
    images2: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np mixup operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images1 (BackendArg): Input `images1` argument (images1).
        images2 (BackendArg): Input `images2` argument (images2).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images1


@global_eager_registry.register("RandAugment")
def _np_rand_augment(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    factor: float,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np rand augment operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images (BackendArg): Input `images` argument (images).
        factor (float): Input `factor` argument (factor).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images


@global_eager_registry.register("RandomColorJitter")
def _np_random_color_jitter(
    backend_module: BackendModuleProtocol | types.ModuleType | None, images: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np random color jitter operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images (BackendArg): Input `images` argument (images).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images


@global_eager_registry.register("RandomCrop")
def _np_random_crop(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    size: BackendArg,
    seed: BackendArg = None,
) -> BackendArg:
    """Execute the eager np random crop operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images (BackendArg): Input `images` argument (images).
        size (BackendArg): Input `size` argument (size).
        seed (BackendArg): Input `seed` argument (seed).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    size_dims = coerce_shape(size)
    size_pair: tuple[int, int] = (size_dims[0], size_dims[1]) if len(size_dims) >= 2 else (1, 1)
    return random_crop_eager(backend_module, images, size_pair, seed)


@global_eager_registry.register("RandomErasing")
def _np_random_erasing(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    factor: float,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np random erasing operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images (BackendArg): Input `images` argument (images).
        factor (float): Input `factor` argument (factor).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return images


def _np_flip_horizontal(img: BackendArg, rng: BackendArg) -> BackendArg:
    """Flip an image horizontally with 50% probability.

    Args:
        img: The image array to flip.
        rng: Random number generator.

    Returns:
        The flipped or unchanged image array.
    """
    rand_val = coerce_float(adapt_backend_module(rng).require_op("random")())
    return coerce_array(_resolve_eager_mod(None), img)[:, :, ::-1] if rand_val > 0.5 else img


def _np_flip_vertical(img: BackendArg, rng: BackendArg) -> BackendArg:
    """Flip an image vertically with 50% probability.

    Args:
        img: The image array to flip.
        rng: Random number generator.

    Returns:
        The flipped or unchanged image array.
    """
    rand_val = coerce_float(adapt_backend_module(rng).require_op("random")())
    return coerce_array(_resolve_eager_mod(None), img)[:, ::-1, :] if rand_val > 0.5 else img


def _np_flip_both(img: BackendArg, rng: BackendArg) -> BackendArg:
    """Flip an image both horizontally and vertically with 50% probability each.

    Args:
        img: The image array to flip.
        rng: Random number generator.

    Returns:
        The flipped or unchanged image array.
    """
    img = _np_flip_horizontal(img, rng)
    return _np_flip_vertical(img, rng)


_NP_FLIP_STRATEGIES: dict[str, Callable[[BackendArg, BackendArg], BackendArg]] = {
    "horizontal": _np_flip_horizontal,
    "vertical": _np_flip_vertical,
    "horizontal_and_vertical": _np_flip_both,
}


def _np_random_flip(images: BackendArg, mode: str, seed: BackendArg = None) -> BackendArg:
    """Execute the eager np random flip operation on the active backend array.

    Args:
        images (BackendArg): Input `images` argument (images).
        mode (str): Input `mode` argument (mode).
        seed (BackendArg): Input `seed` argument (seed).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(None)
    import random

    rng = adapt_backend_module(random).require_symbol("Random")(seed)

    # Check if a single image
    strategy: Callable[[BackendArg, BackendArg], BackendArg] = _NP_FLIP_STRATEGIES.get(mode, lambda img, r: img)
    if len(adapt_backend_module(images).require_symbol("shape")) == 3:
        return strategy(images, rng)

    out = adapt_backend_module(arr_mod).require_symbol("copy")(images)

    for i in range(out.shape[0]):
        out[i] = strategy(out[i], rng)

    return out


class _PositionalVisionModel(pydantic.BaseModel):
    """Base vision model supporting positional and keyword initialization."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Initialize model from positional or keyword arguments.

        Args:
            *args (BackendArg): Positional field values.
            **kwargs (BackendArg): Keyword field values.
        """
        pos_dict: dict[str, BackendArg] = {
            k: args[idx] for idx, k in enumerate(type(self).model_fields.keys()) if idx < len(args)
        }
        super().__init__(**{**pos_dict, **kwargs})


class RotationConfig(_PositionalVisionModel):
    """Rotation configuration."""

    theta: float
    H: int
    W: int
    x: BackendArg
    y: BackendArg


def _calculate_rotation_matrix(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, cfg: RotationConfig
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Calculate rotation matrix coordinates.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): The numpy module.
        cfg (RotationConfig): Rotation configuration.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: src_x, src_y.
    """
    cos_t = adapt_backend_module(arr_mod).require_symbol("cos")(cfg.theta)
    sin_t = adapt_backend_module(arr_mod).require_symbol("sin")(cfg.theta)
    x_num = coerce_numeric(cfg.x)
    y_num = coerce_numeric(cfg.y)

    src_x = adapt_backend_module(arr_mod).require_symbol("asarray")(cos_t * x_num - sin_t * y_num + cfg.W / 2.0)
    src_y = adapt_backend_module(arr_mod).require_symbol("asarray")(sin_t * x_num + cos_t * y_num + cfg.H / 2.0)
    return src_x, src_y


def _nearest_interpolation(
    arr_mod: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    coords: tuple[BackendArg, BackendArg],
    shape: tuple[int, int],
    b_c: tuple[int, int],
) -> BackendArg:
    """Nearest neighbor interpolation.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        images (BackendArg): Input `images` argument (images).
        coords (tuple[BackendArg, BackendArg]): Input `coords` argument (coords).
        shape (tuple[int, int]): Input `shape` argument (shape).
        b_c (tuple[int, int]): Input `b_c` argument (b c).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    src_x, src_y = coords
    H, W = shape
    b, c = b_c
    mod = _resolve_eager_mod(arr_mod)
    img_arr = coerce_array(mod, images)
    src_y_round = mod.clip(mod.round(src_y).astype(mod.int32), 0, H - 1)
    src_x_round = mod.clip(mod.round(src_x).astype(mod.int32), 0, W - 1)
    return img_arr[b, src_y_round, src_x_round, c]


class InterpPixelsConfig(_PositionalVisionModel):
    """InterpPixelsConfig class."""

    images: BackendArg
    y0: BackendArg
    y1: BackendArg
    x0: BackendArg
    x1: BackendArg
    H: int
    W: int
    b_c: tuple[BackendArg, ...]


def _get_interp_pixels(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, cfg: InterpPixelsConfig
) -> tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]:
    """Gather pixels.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        cfg (InterpPixelsConfig): Input `cfg` argument (cfg).

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]: Gathered pixels.
    """
    b, c = cfg.b_c
    img_arr = coerce_array(arr_mod, cfg.images)
    y0_c = adapt_backend_module(arr_mod).require_symbol("clip")(cfg.y0, 0, cfg.H - 1)
    y1_c = adapt_backend_module(arr_mod).require_symbol("clip")(cfg.y1, 0, cfg.H - 1)
    x0_c = adapt_backend_module(arr_mod).require_symbol("clip")(cfg.x0, 0, cfg.W - 1)
    x1_c = adapt_backend_module(arr_mod).require_symbol("clip")(cfg.x1, 0, cfg.W - 1)
    Ia = img_arr[b, y0_c, x0_c, c]
    Ib = img_arr[b, y1_c, x0_c, c]
    Ic = img_arr[b, y0_c, x1_c, c]
    Id = img_arr[b, y1_c, x1_c, c]
    return Ia, Ib, Ic, Id


_NumericWeight = typing.Union[BackendArrayProtocol, float, int]  # noqa: UP007


def _get_interp_weights(
    src_coords: tuple[BackendArg, BackendArg],
    bounds: tuple[BackendArg, BackendArg, BackendArg, BackendArg],
) -> tuple[_NumericWeight, _NumericWeight, _NumericWeight, _NumericWeight]:
    """Execute the eager get interp weights operation on the active backend array.

    Args:
        src_coords (tuple[BackendArg, BackendArg]): Input `src_coords` argument (src coords).
        bounds (tuple[BackendArg, BackendArg, BackendArg, BackendArg]): Input `bounds` argument (bounds).

    Returns:
        tuple[_NumericWeight, _NumericWeight, _NumericWeight, _NumericWeight]: Computed weights.
    """
    src_y = coerce_numeric(src_coords[0])
    src_x = coerce_numeric(src_coords[1])
    y0 = coerce_numeric(bounds[0])
    y1 = coerce_numeric(bounds[1])
    x0 = coerce_numeric(bounds[2])
    x1 = coerce_numeric(bounds[3])
    wa = (y1 - src_y) * (x1 - src_x)
    wb = (src_y - y0) * (x1 - src_x)
    wc = (y1 - src_y) * (src_x - x0)
    wd = (src_y - y0) * (src_x - x0)
    return wa, wb, wc, wd


def _bilinear_interpolation(
    arr_mod: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    coords: tuple[BackendArg, BackendArg],
    shape: tuple[int, int],
    b_c: tuple[int, int],
) -> BackendArg:
    """Bilinear interpolation.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        images (BackendArg): Input `images` argument (images).
        coords (tuple[BackendArg, BackendArg]): Input `coords` argument (coords).
        shape (tuple[int, int]): Input `shape` argument (shape).
        b_c (tuple[int, int]): Input `b_c` argument (b c).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    mod = _resolve_eager_mod(arr_mod)
    y0: EagerTensorProtocol = mod.floor(coords[1]).astype(mod.int32)
    y1: EagerTensorProtocol = y0 + 1
    x0: EagerTensorProtocol = mod.floor(coords[0]).astype(mod.int32)
    x1: EagerTensorProtocol = x0 + 1

    pixel_cfg = InterpPixelsConfig(images=images, y0=y0, y1=y1, x0=x0, x1=x1, H=shape[0], W=shape[1], b_c=b_c)
    p0, p1, p2, p3 = _get_interp_pixels(mod, pixel_cfg)
    w0, w1, w2, w3 = _get_interp_weights((coords[1], coords[0]), (y0, y1, x0, x1))
    return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3


class AffineConfig(_PositionalVisionModel):
    """Affine configuration."""

    coords: tuple[BackendArg, ...]
    shape: tuple[BackendArg, ...]
    b_c: tuple[BackendArg, ...]
    options: tuple[BackendArg, ...]


def _apply_affine_grid(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, images: BackendArg, cfg: AffineConfig
) -> BackendArg:
    """Apply affine grid sampling.

    Args:
        arr_mod: The numpy module.
        images: Input images.
        cfg: Affine configuration.

    Returns: EagerTensorProtocol: Sampled values.
    """
    interpolation, fill_mode, fill_value, mask = cfg.options
    coords_pair = (cfg.coords[0], cfg.coords[1])
    shape_pair = (int(coerce_int(cfg.shape[0])), int(coerce_int(cfg.shape[1])))
    bc_pair = (int(coerce_int(cfg.b_c[0])), int(coerce_int(cfg.b_c[1])))
    if interpolation == "nearest":
        val = _nearest_interpolation(arr_mod, images, coords_pair, shape_pair, bc_pair)
    else:
        val = _bilinear_interpolation(arr_mod, images, coords_pair, shape_pair, bc_pair)

    if fill_mode == "constant":
        val = adapt_backend_module(arr_mod).require_symbol("where")(mask, val, fill_value)

    return adapt_backend_module(val).require_symbol("astype")(adapt_backend_module(images).require_symbol("dtype"))


class BatchRotationConfig(_PositionalVisionModel):
    """Batch rotation configuration."""

    images: EagerTensorProtocol
    angles: EagerTensorProtocol
    H: int
    W: int
    x: BackendArg
    y: BackendArg
    options: tuple[BackendArg, ...]


def _process_batch_item(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, cfg: BatchRotationConfig, b: int, out: BackendArg
) -> None:
    """Process a single item in the batch.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        cfg (BatchRotationConfig): Input `cfg` argument (cfg).
        b (int): Input `b` argument (b).
        out (BackendArg): Input `out` argument (out).
    """
    out_arr = coerce_array(arr_mod, out)
    rot_cfg = RotationConfig(theta=coerce_float(cfg.angles[b]), H=cfg.H, W=cfg.W, x=cfg.x, y=cfg.y)
    src_x, src_y = _calculate_rotation_matrix(arr_mod, rot_cfg)
    mask = (src_x >= 0) & (src_x <= cfg.W - 1) & (src_y >= 0) & (src_y <= cfg.H - 1)
    C = int(tuple(cfg.images.shape)[3])
    for c in range(C):
        aff_cfg = AffineConfig(coords=(src_x, src_y), shape=(cfg.H, cfg.W), b_c=(b, c), options=cfg.options + (mask,))
        out_arr[b, :, :, c] = _apply_affine_grid(arr_mod, cfg.images, aff_cfg)


def _apply_rotation_batch(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, cfg: BatchRotationConfig
) -> EagerTensorProtocol:
    """Apply rotation to a batch of images.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        cfg (BatchRotationConfig): Input `cfg` argument (cfg).

    Returns:
            EagerTensorProtocol: Rotated batch tensor.
    """
    B = int(tuple(cfg.images.shape)[0])
    out = adapt_backend_module(arr_mod).require_symbol("zeros_like")(cfg.images, dtype=cfg.images.dtype)
    for b in range(B):
        _process_batch_item(arr_mod, cfg, b, out)
    return out


def _resolve_rotation_factor(
    factor: BackendArg,
) -> tuple[float, float]:
    """Resolve rotation bounds.

    Args:
        factor (BackendArg): Input `factor` argument (factor).

    Returns:
        tuple[float, float]: Lower and upper rotation bounds.
    """
    if isinstance(factor, (list, tuple)):
        return coerce_float(factor[0]), coerce_float(factor[1])
    f_val = coerce_float(factor)
    return -f_val, f_val


def _create_rotation_mesh(arr_mod: BackendModuleProtocol | types.ModuleType | None, H: int, W: int) -> BackendArg:
    """Create mesh grid for rotation.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        H (int): Input `H` argument (H).
        W (int): Input `W` argument (W).

    Returns:
        tuple: Computed `tuple` output.
    """
    y, x = adapt_backend_module(arr_mod).require_symbol("meshgrid")(
        adapt_backend_module(arr_mod).require_symbol("arange")(H),
        adapt_backend_module(arr_mod).require_symbol("arange")(W),
        indexing="ij",
    )
    y = y.astype(adapt_backend_module(arr_mod).require_symbol("float32")) - H / 2.0
    x = x.astype(adapt_backend_module(arr_mod).require_symbol("float32")) - W / 2.0
    return y, x


@global_eager_registry.register("RandomRotation")
def _np_random_rotation(
    backend_module: BackendModuleProtocol | types.ModuleType | None, images: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np random rotation operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images (BackendArg): Input `images` argument (images).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    images = arr_mod.asarray(images)
    rank = len(adapt_backend_module(images).require_symbol("shape"))
    if rank == 3:
        images = arr_mod.expand_dims(images, 0)
    elif rank != 4:
        return images

    raw_b, raw_h, raw_w, _ = adapt_backend_module(images).require_symbol("shape")
    B, H, W = coerce_int(raw_b), coerce_int(raw_h), coerce_int(raw_w)

    rng = arr_mod.random.default_rng(kwargs.get("seed", None))
    angles = rng.uniform(*_resolve_rotation_factor(kwargs.get("factor", 0.0)), size=(B,)) * 2 * arr_mod.pi

    y, x = unpack_pair(_create_rotation_mesh(arr_mod, H, W))

    options = (
        kwargs.get("interpolation", "bilinear"),
        kwargs.get("fill_mode", "reflect"),
        kwargs.get("fill_value", 0.0),
    )

    batch_cfg = BatchRotationConfig(images=images, angles=angles, H=H, W=W, x=x, y=y, options=options)
    out = _apply_rotation_batch(arr_mod, batch_cfg)

    return out[0] if rank == 3 else out


@global_eager_registry.register("RandomTranslation")
def _np_random_translation(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    height_factor: BackendArg,
    width_factor: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np random translation operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images (BackendArg): Input `images` argument (images).
        height_factor (BackendArg): Input `height_factor` argument (height factor).
        width_factor (BackendArg): Input `width_factor` argument (width factor).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    hf: tuple[float, float] | float = (
        (coerce_float(height_factor[0]), coerce_float(height_factor[1]))
        if isinstance(height_factor, (list, tuple)) and len(height_factor) >= 2
        else coerce_float(height_factor)
    )
    wf: tuple[float, float] | float = (
        (coerce_float(width_factor[0]), coerce_float(width_factor[1]))
        if isinstance(width_factor, (list, tuple)) and len(width_factor) >= 2
        else coerce_float(width_factor)
    )
    return random_translation_eager(backend_module, images, hf, wf, **kwargs)


@global_eager_registry.register("RandomZoom")
def _np_random_zoom(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    height_factor: BackendArg,
    width_factor: BackendArg = None,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np random zoom operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        images (BackendArg): Input `images` argument (images).
        height_factor (BackendArg): Input `height_factor` argument (height factor).
        width_factor (BackendArg): Input `width_factor` argument (width factor).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    hf: tuple[float, float] | float = (
        (coerce_float(height_factor[0]), coerce_float(height_factor[1]))
        if isinstance(height_factor, (list, tuple)) and len(height_factor) >= 2
        else coerce_float(height_factor)
    )
    wf: tuple[float, float] | float | None = (
        None
        if width_factor is None
        else (
            (coerce_float(width_factor[0]), coerce_float(width_factor[1]))
            if isinstance(width_factor, (list, tuple)) and len(width_factor) >= 2
            else coerce_float(width_factor)
        )
    )
    return random_zoom_eager(backend_module, images, hf, wf, **kwargs)


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
    "global_eager_registry",
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
    "global_eager_registry",
]
