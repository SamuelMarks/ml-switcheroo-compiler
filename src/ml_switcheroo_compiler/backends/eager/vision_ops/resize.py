"""Generic backend-parametric eager reference operations for `vision_ops.resize`."""

from __future__ import annotations

import types
import typing  # noqa: F401

import pydantic

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_float,
    coerce_int,
    coerce_int_list,
    global_eager_registry,
    unpack_pair,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


class ResizeConfig(pydantic.BaseModel):
    """Configuration for resize operations.

    Attributes:
        new_H: New height.
        new_W: New width.
        H: Old height.
        W: Old width.
        align_corners: Align corners.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    new_H: int
    new_W: int
    H: int
    W: int
    align_corners: bool


class InterpolationConfig(pydantic.BaseModel):
    """Configuration for interpolation.

    Attributes:
        images: Input images tensor.
        src_y: Source y coordinates.
        src_x: Source x coordinates.
        H: Original height.
        W: Original width.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    images: EagerTensorProtocol | None = None
    src_y: EagerTensorProtocol | None = None
    src_x: EagerTensorProtocol | None = None
    H: int = 0
    W: int = 0


def _calculate_bilinear_coords(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, cfg: ResizeConfig
) -> BackendArg:
    """Calculate coords for bilinear resize.

    Args:
        arr_mod: Numpy module.
        cfg: Resize configuration.

    Returns:
        tuple: src_y, src_x.
    """
    y = adapt_backend_module(arr_mod).require_symbol("arange")(
        cfg.new_H, dtype=adapt_backend_module(arr_mod).require_symbol("float32")
    )
    x = adapt_backend_module(arr_mod).require_symbol("arange")(
        cfg.new_W, dtype=adapt_backend_module(arr_mod).require_symbol("float32")
    )

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

    return adapt_backend_module(arr_mod).require_symbol("clip")(src_y, 0, cfg.H - 1), adapt_backend_module(
        arr_mod
    ).require_symbol("clip")(src_x, 0, cfg.W - 1)


class BilinearCoords(pydantic.BaseModel):
    """Coordinates for bilinear interpolation."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    y0: EagerTensorProtocol | EagerTensorProtocol
    y1: EagerTensorProtocol | EagerTensorProtocol
    x0: EagerTensorProtocol | EagerTensorProtocol
    x1: EagerTensorProtocol | EagerTensorProtocol


def _compute_bilinear_pixels(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, images: BackendArg, coords: BilinearCoords
) -> tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]:
    """Fetch pixels for bilinear interpolation.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Numpy module.
        images (BackendArg): Input images.
        coords (BilinearCoords): Tuple of (y0, y1, x0, x1).

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]: Four corner pixels.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    img_arr = coerce_array(arr_mod, images)
    y0, y1, x0, x1 = coords.y0, coords.y1, coords.x0, coords.x1
    Ia = img_arr[:, y0[:, None], x0[None, :], :]
    Ib = img_arr[:, y1[:, None], x0[None, :], :]
    Ic = img_arr[:, y0[:, None], x1[None, :], :]
    Id = img_arr[:, y1[:, None], x1[None, :], :]
    return Ia, Ib, Ic, Id


def _compute_bilinear_weights(
    dy: EagerTensorProtocol | EagerTensorProtocol,
    dx: EagerTensorProtocol | EagerTensorProtocol,
) -> tuple[
    EagerTensorProtocol | EagerTensorProtocol,
    EagerTensorProtocol | EagerTensorProtocol,
    EagerTensorProtocol | EagerTensorProtocol,
    EagerTensorProtocol | EagerTensorProtocol,
]:
    """Execute the eager compute bilinear weights operation on the active backend array.

    Args:
        dy (EagerTensorProtocol | EagerTensorProtocol): Input `dy` argument (dy).
        dx (EagerTensorProtocol | EagerTensorProtocol): Input `dx` argument (dx).

    Returns:
        tuple: Computed `tuple` output.
    """
    wa = (1 - dy[:, None, None]) * (1 - dx[None, :, None])
    wb = (dy[:, None, None]) * (1 - dx[None, :, None])
    wc = (1 - dy[:, None, None]) * (dx[None, :, None])
    wd = (dy[:, None, None]) * (dx[None, :, None])
    return wa, wb, wc, wd


def _apply_bilinear_interpolation(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, cfg: InterpolationConfig
) -> BackendArg:
    """Apply bilinear interpolation.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        cfg (InterpolationConfig): Input `cfg` argument (cfg).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    assert cfg.src_y is not None and cfg.src_x is not None and cfg.images is not None
    y0 = (
        adapt_backend_module(arr_mod)
        .require_symbol("floor")(cfg.src_y)
        .astype(adapt_backend_module(arr_mod).require_symbol("int32"))
    )
    y1 = adapt_backend_module(arr_mod).require_symbol("clip")(y0 + 1, 0, cfg.H - 1)
    x0 = (
        adapt_backend_module(arr_mod)
        .require_symbol("floor")(cfg.src_x)
        .astype(adapt_backend_module(arr_mod).require_symbol("int32"))
    )
    x1 = adapt_backend_module(arr_mod).require_symbol("clip")(x0 + 1, 0, cfg.W - 1)

    dy = cfg.src_y - y0
    dx = cfg.src_x - x0

    pixels = _compute_bilinear_pixels(arr_mod, cfg.images, BilinearCoords(y0=y0, y1=y1, x0=x0, x1=x1))
    weights = _compute_bilinear_weights(dy, dx)

    p0, p1, p2, p3 = pixels
    w0, w1, w2, w3 = weights
    blended = p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3
    return adapt_backend_module(blended).require_op("astype")(adapt_backend_module(cfg.images).require_symbol("dtype"))


@global_eager_registry.register("ResizeBilinear")
def resize_bilinear(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    size: tuple[int, int],
    align_corners: bool = False,
) -> BackendArg:
    """Resize images using bilinear interpolation.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        images (BackendArg): Input `images` argument (images).
        size (tuple[int, int]): Input `size` argument (size).
        align_corners (bool): Input `align_corners` argument (align corners).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    images = adapt_backend_module(np_mod).require_symbol("asarray")(images)
    rank = len(adapt_backend_module(images).require_symbol("shape"))
    if rank == 3:
        images = adapt_backend_module(np_mod).require_symbol("expand_dims")(images, 0)
    elif rank != 4:
        return images

    shape = adapt_backend_module(images).require_symbol("shape")
    h_int, w_int = coerce_int(shape[1]), coerce_int(shape[2])
    resize_cfg = ResizeConfig(new_H=size[0], new_W=size[1], H=h_int, W=w_int, align_corners=align_corners)
    src_y, src_x = unpack_pair(_calculate_bilinear_coords(np_mod, resize_cfg))

    interp_cfg = InterpolationConfig(
        images=_ensure_arr(images), src_y=_ensure_arr(src_y), src_x=_ensure_arr(src_x), H=h_int, W=w_int
    )
    out = _ensure_arr(_apply_bilinear_interpolation(np_mod, interp_cfg))

    return out[0] if rank == 3 else out


def _calculate_nearest_coords(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, cfg: ResizeConfig
) -> BackendArg:
    """Calculate coords for nearest resize.

    Args:
        arr_mod: Numpy module.
        cfg: Resize configuration.

    Returns:
        tuple: src_y, src_x.
    """
    y = adapt_backend_module(arr_mod).require_symbol("arange")(
        cfg.new_H, dtype=adapt_backend_module(arr_mod).require_symbol("float32")
    )
    x = adapt_backend_module(arr_mod).require_symbol("arange")(
        cfg.new_W, dtype=adapt_backend_module(arr_mod).require_symbol("float32")
    )

    if cfg.align_corners:
        y_scale = (cfg.H - 1) / (cfg.new_H - 1) if cfg.new_H > 1 else 0
        x_scale = (cfg.W - 1) / (cfg.new_W - 1) if cfg.new_W > 1 else 0
        src_y = (
            adapt_backend_module(arr_mod)
            .require_symbol("round")(y * y_scale)
            .astype(adapt_backend_module(arr_mod).require_symbol("int32"))
        )
        src_x = (
            adapt_backend_module(arr_mod)
            .require_symbol("round")(x * x_scale)
            .astype(adapt_backend_module(arr_mod).require_symbol("int32"))
        )
    else:
        y_scale = cfg.H / cfg.new_H
        x_scale = cfg.W / cfg.new_W
        src_y = (
            adapt_backend_module(arr_mod)
            .require_symbol("floor")(y * y_scale)
            .astype(adapt_backend_module(arr_mod).require_symbol("int32"))
        )
        src_x = (
            adapt_backend_module(arr_mod)
            .require_symbol("floor")(x * x_scale)
            .astype(adapt_backend_module(arr_mod).require_symbol("int32"))
        )

    return adapt_backend_module(arr_mod).require_symbol("clip")(src_y, 0, cfg.H - 1), adapt_backend_module(
        arr_mod
    ).require_symbol("clip")(src_x, 0, cfg.W - 1)


@global_eager_registry.register("ResizeNearest")
def resize_nearest(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    size: tuple[int, int],
    align_corners: bool = False,
) -> BackendArg:
    """Resize images using nearest-neighbor interpolation.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        images (BackendArg): Input `images` argument (images).
        size (tuple[int, int]): Input `size` argument (size).
        align_corners (bool): Input `align_corners` argument (align corners).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    images = adapt_backend_module(np_mod).require_symbol("asarray")(images)
    rank = len(adapt_backend_module(images).require_symbol("shape"))
    if rank == 3:
        images = adapt_backend_module(np_mod).require_symbol("expand_dims")(images, 0)
    elif rank != 4:
        return images

    shape = adapt_backend_module(images).require_symbol("shape")
    resize_cfg = ResizeConfig(
        new_H=size[0], new_W=size[1], H=coerce_int(shape[1]), W=coerce_int(shape[2]), align_corners=align_corners
    )
    raw_sy, raw_sx = unpack_pair(_calculate_nearest_coords(np_mod, resize_cfg))
    src_y, src_x = _ensure_arr(raw_sy), _ensure_arr(raw_sx)

    out = images[:, src_y[:, None], src_x[None, :], :]

    return out[0] if rank == 3 else out


@global_eager_registry.register("ResizeBicubic")
def resize_bicubic(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    size: tuple[int, int],
    align_corners: bool = False,
) -> BackendArg:
    """Resize images using bicubic interpolation.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        images (BackendArg): Input `images` argument (images).
        size (tuple[int, int]): Input `size` argument (size).
        align_corners (bool): Input `align_corners` argument (align corners).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return resize_bilinear(np_mod, images, size, align_corners)


@global_eager_registry.register("ResizeLanczos3")
def resize_lanczos3(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    size: tuple[int, int],
    align_corners: bool = False,
) -> BackendArg:
    """Resize images using Lanczos3 interpolation.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        images (BackendArg): Input `images` argument (images).
        size (tuple[int, int]): Input `size` argument (size).
        align_corners (bool): Input `align_corners` argument (align corners).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return resize_bilinear(np_mod, images, size, align_corners)


@global_eager_registry.register("RandomFlip")
def random_flip_numpy(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    mode: BackendArg = "horizontal_and_vertical",
    seed: BackendArg = None,
    **kwargs: BackendArg,
) -> BackendArg:
    """Generate random flip.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        images (BackendArg): Input `images` argument (images).
        mode (BackendArg): Input `mode` argument (mode).
        seed (BackendArg): Input `seed` argument (seed).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(np_mod)
    x = arr_mod.asarray(images)
    rank = len(x.shape)
    if rank == 3:
        x = arr_mod.expand_dims(x, 0)
    elif rank != 4:
        return x

    rng = arr_mod.random.default_rng(seed)
    out = arr_mod.copy(x)

    if mode in ("horizontal", "horizontal_and_vertical"):
        mask_h = rng.random(size=(x.shape[0],)) > 0.5
        out[mask_h] = out[mask_h, :, ::-1, :]

    if mode in ("vertical", "horizontal_and_vertical"):
        mask_v = rng.random(size=(x.shape[0],)) > 0.5
        out[mask_v] = out[mask_v, ::-1, :, :]

    if rank == 3:
        return out[0]
    return out


def _compute_rotation_coords(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, H: int, W: int, angle: float
) -> BackendArg:
    """Execute the eager compute rotation coords operation on the active backend array.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        H (int): Input `H` argument (H).
        W (int): Input `W` argument (W).
        angle (float): Input `angle` argument (angle).

    Returns:
        tuple: Computed `tuple` output.
    """
    cos_a = adapt_backend_module(arr_mod).require_symbol("cos")(angle)
    sin_a = adapt_backend_module(arr_mod).require_symbol("sin")(angle)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    y_coords, x_coords = (
        adapt_backend_module(arr_mod)
        .require_symbol("mgrid")[0:H, 0:W]
        .astype(adapt_backend_module(arr_mod).require_symbol("float32"))
    )
    x_shifted = x_coords - cx
    y_shifted = y_coords - cy
    src_x = x_shifted * cos_a + y_shifted * sin_a + cx
    src_y = -x_shifted * sin_a + y_shifted * cos_a + cy
    return src_x, src_y


class RotationInterpolationConfig(pydantic.BaseModel):
    """Configuration for rotation interpolation.

    Attributes:
        img: Input image.
        src_x: X coordinates.
        src_y: Y coordinates.
        fill_mode: Fill mode.
        fill_value: Fill value.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    img: EagerTensorProtocol | None = None
    src_x: EagerTensorProtocol | None = None
    src_y: EagerTensorProtocol | None = None
    fill_mode: str = "nearest"
    fill_value: float = 0.0


def _get_rotation_coords(
    arr_mod: BackendModuleProtocol | types.ModuleType | None,
    src_x_clip: BackendArg,
    src_y_clip: BackendArg,
    W: int,
    H: int,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]:
    """Get the bounding coordinates for rotation interpolation.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): The numpy module.
        src_x_clip (BackendArg): Clipped source X coordinates.
        src_y_clip (BackendArg): Clipped source Y coordinates.
        W (int): Image width.
        H (int): Image height.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]:
            (x0, x1, y0, y1) coordinates.
    """
    x0 = (
        adapt_backend_module(arr_mod)
        .require_symbol("floor")(src_x_clip)
        .astype(adapt_backend_module(arr_mod).require_symbol("int32"))
    )
    x1 = adapt_backend_module(arr_mod).require_symbol("clip")(x0 + 1, 0, W - 1)
    y0 = (
        adapt_backend_module(arr_mod)
        .require_symbol("floor")(src_y_clip)
        .astype(adapt_backend_module(arr_mod).require_symbol("int32"))
    )
    y1 = adapt_backend_module(arr_mod).require_symbol("clip")(y0 + 1, 0, H - 1)
    return x0, x1, y0, y1


def _gather_rotation_pixels(
    img: EagerTensorProtocol,
    coords: tuple[
        EagerTensorProtocol,
        EagerTensorProtocol,
        EagerTensorProtocol,
        EagerTensorProtocol,
    ],
) -> tuple[
    EagerTensorProtocol,
    EagerTensorProtocol,
    EagerTensorProtocol,
    EagerTensorProtocol,
]:
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
    dx: EagerTensorProtocol,
    dy: EagerTensorProtocol,
) -> tuple[
    EagerTensorProtocol,
    EagerTensorProtocol,
    EagerTensorProtocol,
    EagerTensorProtocol,
]:
    """Execute the eager compute rotation weights operation on the active backend array.

    Args:
        dx (EagerTensorProtocol): Input `dx` argument (dx).
        dy (EagerTensorProtocol): Input `dy` argument (dy).

    Returns:
        tuple: Computed `tuple` output.
    """
    wa = (1 - dy)[..., None] * (1 - dx)[..., None]
    wb = dy[..., None] * (1 - dx)[..., None]
    wc = (1 - dy)[..., None] * dx[..., None]
    wd = dy[..., None] * dx[..., None]
    return wa, wb, wc, wd


def _interpolate_rotation(
    arr_mod: BackendModuleProtocol | types.ModuleType | None, cfg: RotationInterpolationConfig
) -> BackendArg:
    """Interpolate rotated image.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        cfg (RotationInterpolationConfig): Input `cfg` argument (cfg).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    assert cfg.img is not None
    raw_h, raw_w, _ = adapt_backend_module(cfg.img).require_symbol("shape")
    H, W = coerce_int(raw_h), coerce_int(raw_w)
    src_x_clip = adapt_backend_module(arr_mod).require_symbol("clip")(cfg.src_x, 0, W - 1)
    src_y_clip = adapt_backend_module(arr_mod).require_symbol("clip")(cfg.src_y, 0, H - 1)

    coords = _get_rotation_coords(arr_mod, src_x_clip, src_y_clip, W, H)
    dx = _ensure_arr(src_x_clip - coords[0])
    dy = _ensure_arr(src_y_clip - coords[2])

    pixels = _gather_rotation_pixels(cfg.img, coords)
    weights = _compute_rotation_weights(dx, dy)

    interp = (
        adapt_backend_module(arr_mod).require_symbol("multiply")(pixels[0], weights[0])
        + adapt_backend_module(arr_mod).require_symbol("multiply")(pixels[1], weights[1])
        + adapt_backend_module(arr_mod).require_symbol("multiply")(pixels[2], weights[2])
        + adapt_backend_module(arr_mod).require_symbol("multiply")(pixels[3], weights[3])
    )
    valid_mask = (
        (adapt_backend_module(arr_mod).require_symbol("asarray")(cfg.src_x, dtype=float) >= 0)
        & (adapt_backend_module(arr_mod).require_symbol("asarray")(cfg.src_x, dtype=float) <= W - 1)
        & (adapt_backend_module(arr_mod).require_symbol("asarray")(cfg.src_y, dtype=float) >= 0)
        & (adapt_backend_module(arr_mod).require_symbol("asarray")(cfg.src_y, dtype=float) <= H - 1)
    )
    if cfg.fill_mode == "constant":
        interp[~valid_mask] = cfg.fill_value
    return interp.astype(adapt_backend_module(cfg.img).require_symbol("dtype"))


def _apply_random_rotation_single(
    arr_mod: BackendModuleProtocol | types.ModuleType | None,
    img: BackendArg,
    angle: float,
    fill_mode: str,
    fill_value: float,
) -> BackendArg:
    """Apply random rotation to a single image.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        img (BackendArg): Input `img` argument (img).
        angle (float): Input `angle` argument (angle).
        fill_mode (str): Input `fill_mode` argument (fill mode).
        fill_value (float): Input `fill_value` argument (fill value).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    H, W, _ = coerce_int_list(adapt_backend_module(img).require_symbol("shape"))
    src_x, src_y = unpack_pair(_compute_rotation_coords(arr_mod, H, W, angle))
    cfg = RotationInterpolationConfig(
        img=_ensure_arr(img),
        src_x=_ensure_arr(src_x),
        src_y=_ensure_arr(src_y),
        fill_mode=fill_mode,
        fill_value=fill_value,
    )
    return _interpolate_rotation(arr_mod, cfg)


class TransformOptions(pydantic.BaseModel):
    """Options for image transformations."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    fill_mode: str = "reflect"
    interpolation: str = "bilinear"
    seed: int | None = None
    fill_value: float = 0.0


def _apply_rotation_to_batch(
    arr_mod: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    rng: BackendArg,
    angles: tuple[float, float],
    options: TransformOptions,
) -> BackendArg:
    """Apply rotation to batch of images.

    Args:
        arr_mod (BackendModuleProtocol | types.ModuleType | None): Input `np` argument (np).
        x (BackendArg): Input `x` argument (x).
        rng (BackendArg): Input `rng` argument (rng).
        angles (tuple[float, float]): Input `angles` argument (angles).
        options (TransformOptions): Input `options` argument (options).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    x_arr = coerce_array(arr_mod, x)
    B = x_arr.shape[0]
    out = adapt_backend_module(arr_mod).require_symbol("zeros_like")(x_arr)
    for i in range(B):
        angle = (
            float(coerce_float(adapt_backend_module(rng).require_symbol("uniform")(angles[0], angles[1])))
            * 2.0
            * float(adapt_backend_module(arr_mod).require_symbol("pi"))
        )
        out[i] = _apply_random_rotation_single(arr_mod, x_arr[i], angle, options.fill_mode, options.fill_value)
    return out


@global_eager_registry.register("RandomRotation")
def random_rotation_numpy(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    factor: float | tuple[float, float] | list[float],
    options: TransformOptions | None = None,
    **kwargs: BackendArg,
) -> BackendArg:
    """Generate random rotation.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        images (BackendArg): Input `images` argument (images).
        factor (float | tuple[float, float] | list[float]): Input `factor` argument (factor).
        options (TransformOptions | None): Input `options` argument (options).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(np_mod)
    options = options or TransformOptions()
    x = arr_mod.asarray(images)
    rank = len(x.shape)
    if rank == 3:
        x = arr_mod.expand_dims(x, 0)
    elif rank != 4:
        return x

    rng = arr_mod.random.default_rng(options.seed)

    if isinstance(factor, (tuple, list)):
        lower, upper = factor
    else:
        lower, upper = -factor, factor

    out = _ensure_arr(_apply_rotation_to_batch(arr_mod, x, rng, (lower, upper), options))

    if rank == 3:
        return out[0]
    return out
