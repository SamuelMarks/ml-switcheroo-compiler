"""Compiler `vision augmentation` operations and primitives for the `eager` subsystem."""

from __future__ import annotations

import types
from collections.abc import Callable

import pydantic

from ml_switcheroo_compiler.backends.eager.utils import _from_channels_last, _from_numpy_array
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_array,
    coerce_float,
    coerce_opt_str,
    coerce_str,
    unpack_pair,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.shape import OpValue as OpValue

from .vision_transforms import _apply_elastic_batch
from .vision_utils import (
    GeometricGridConfig,
    RandomCropConfig,
    TransformInterpolationConfig,
    _np_map_coordinates,
    _prepare_eager_transform,
)

VisionArray = BackendArg


def _flip_horizontal(img: VisionArray, rng: VisionArray) -> VisionArray:
    """Flip an image horizontally with 50% probability.

    Args:
        img: The input image array to flip.
        rng: The random number generator instance.

    Returns: VisionArray: The horizontally flipped image, or the original image.
    """
    from .vision_utils import _resolve_backend_mod

    rand_val = coerce_float(adapt_backend_module(rng).require_op("random")())
    return coerce_array(_resolve_backend_mod(None), img)[:, :, ::-1] if rand_val > 0.5 else img


def _flip_vertical(img: VisionArray, rng: VisionArray) -> VisionArray:
    """Flip an image vertically with 50% probability.

    Args:
        img: The input image array to flip.
        rng: The random number generator instance.

    Returns: VisionArray: The vertically flipped image, or the original image.
    """
    from .vision_utils import _resolve_backend_mod

    rand_val = coerce_float(adapt_backend_module(rng).require_op("random")())
    return coerce_array(_resolve_backend_mod(None), img)[:, ::-1, :] if rand_val > 0.5 else img


def _flip_both(img: VisionArray, rng: VisionArray) -> VisionArray:
    """Flip an image both horizontally and vertically with 50% probability each.

    Args:
        img: The input image array to flip.
        rng: The random number generator instance.

    Returns: VisionArray: The flipped image, potentially along both axes.
    """
    img = _flip_horizontal(img, rng)
    return _flip_vertical(img, rng)


def _identity_flip(img: VisionArray, rng: VisionArray) -> VisionArray:
    """Return the image unchanged.

    Args:
        img: The input image array.
        rng: The random number generator instance.

    Returns:
        VisionArray: The unchanged image.
    """
    del rng
    return img


_FLIP_STRATEGIES: dict[str, Callable[[VisionArray, VisionArray], VisionArray]] = {
    "horizontal": _flip_horizontal,
    "vertical": _flip_vertical,
    "horizontal_and_vertical": _flip_both,
}


def random_flip_eager(
    backend_module: types.ModuleType | BackendModuleProtocol | None,
    images: VisionArray,
    mode: str,
    seed: VisionArray | None = None,
) -> VisionArray:
    """Apply a random flip transformation to a batch of images eagerly.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        images: Input `images` argument (images).
        mode (str): Input `mode` argument (mode).
        seed: Input `seed` argument (seed).

    Returns:
            VisionArray: Computed `VisionArray` output.
    """
    from .vision_utils import _resolve_backend_mod

    data_format = None
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)
    np_mod = _resolve_backend_mod(ctx.np_mod)

    out = adapt_backend_module(np_mod).require_symbol("copy")(ctx.imgs)
    strategy = _FLIP_STRATEGIES.get(mode, _identity_flip)

    for i in range(ctx.B):
        out[i] = strategy(out[i], ctx.rng)

    out_cl = _from_channels_last(np_mod, out, data_format)
    return _from_numpy_array(backend_module, out_cl, "", images)


class RotationConfig(pydantic.BaseModel):
    """Configuration parameters for the random rotation transformation."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    factor: float | tuple[float, float] = 0.0
    fill_mode: str = "reflect"
    interpolation: str = "bilinear"
    fill_value: float = 0.0
    data_format: str | None = None
    seed: VisionArray | None = None


def _compute_rotation_matrix(np_mod: VisionArray, angle: float, W: int, H: int) -> tuple[float, float]:
    """Calculate the 2D affine matrix coefficients for a given rotation angle.

    Args:
        np_mod: The numpy-like module for math operations.
        angle (float): The rotation angle in radians.
        W (int): The width of the image bounding box.
        H (int): The height of the image bounding box.

    Returns:
        tuple[float, float]: The sine and cosine coefficients for the transform.
    """
    return (0.0, 0.0)


def _generate_coordinate_grid(
    np_mod: types.ModuleType | BackendModuleProtocol, H: int, W: int
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Create a 2D meshgrid corresponding to image coordinates.

    Args:
        np_mod: The numpy-like module for array creation.
        H (int): The height dimension of the grid.
        W (int): The width dimension of the grid.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: The Y and X coordinate grids.
    """
    from .vision_utils import _resolve_backend_mod

    mod = _resolve_backend_mod(np_mod)
    ys, xs = unpack_pair(mod.meshgrid(mod.arange(H, dtype=float), mod.arange(W, dtype=float), indexing="ij"))
    return (mod.asarray(ys), mod.asarray(xs))


class AffineTransformParams(pydantic.BaseModel):
    """Parameters defining a spatial affine transformation operation."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    cos_a: float
    sin_a: float
    cx: float
    cy: float


def _apply_affine_transform(
    y_grid: VisionArray, x_grid: VisionArray, params: AffineTransformParams
) -> tuple[VisionArray, VisionArray]:
    """Transform spatial coordinate grids using an affine matrix.

    Args:
        y_grid: The Y-coordinate grid to transform.
        x_grid: The X-coordinate grid to transform.
        params (AffineTransformParams): The parameters of the affine transformation.

    Returns:
        tuple[VisionArray, VisionArray]: The transformed Y and X coordinates.
    """
    x_shifted = coerce_float(x_grid) - params.cx
    y_shifted = coerce_float(y_grid) - params.cy
    x_rot = x_shifted * params.cos_a + y_shifted * params.sin_a
    y_rot = -x_shifted * params.sin_a + y_shifted * params.cos_a
    return (y_rot + params.cy, x_rot + params.cx)


def _interpolate_pixels(
    np_mod: types.ModuleType | BackendModuleProtocol,
    imgs: VisionArray,
    new_y: VisionArray,
    new_x: VisionArray,
    config: RotationConfig,
) -> EagerTensorProtocol:
    """Sample pixels from original images at new spatial coordinates using interpolation.

    Args:
        np_mod: The numpy-like module for array manipulation.
        imgs: The batch of source images.
        new_y: The Y-coordinates to sample from.
        new_x: The X-coordinates to sample from.
        config (RotationConfig): The configuration dictating interpolation method and out-of-bounds behavior.

    Returns: EagerTensorProtocol: The interpolated image tensor.
    """
    from .vision_utils import _resolve_backend_mod

    mod = _resolve_backend_mod(np_mod)
    imgs_t = coerce_array(mod, imgs)
    b_dim, h_dim, w_dim, c_dim = imgs_t.shape
    ny = mod.broadcast_to(mod.asarray(new_y, dtype=float), (h_dim, w_dim))
    nx = mod.broadcast_to(mod.asarray(new_x, dtype=float), (h_dim, w_dim))
    out = mod.zeros_like(imgs_t)
    order = 1 if config.interpolation == "bilinear" else 0
    for b in range(b_dim):
        for c in range(c_dim):
            out[b, :, :, c] = _np_map_coordinates(
                mod,
                imgs_t[b, :, :, c],
                (ny, nx),
                order=order,
                fill_value=config.fill_value,
            )
    return out


def _compute_rotation_grid(
    np_mod: VisionArray, H: int, W: int, rng: VisionArray, factor: float
) -> tuple[VisionArray, VisionArray]:
    """Calculate the transformed spatial grid for a random rotation.

    Args:
        np_mod: The numpy-like module for array operations.
        H (int): The height of the spatial grid.
        W (int): The width of the spatial grid.
        rng: The random number generator instance.
        factor (float): The maximum rotation factor in radians.

    Returns:
        tuple[VisionArray, VisionArray]: The Y and X coordinates of the rotated grid.
    """
    return (0, 0)


def random_rotation_eager(backend_module: VisionArray, images: VisionArray, config: RotationConfig) -> VisionArray:
    """Apply a random rotation transformation to an image batch eagerly.

    Args:
        backend_module: The backend module to use for array operations.
        images: The input tensor of images to rotate.
        config (RotationConfig): The rotation configuration parameters.

    Returns: VisionArray: The batch of rotated images.
    """
    return (0, 0)


def _crop_and_pad_single(
    np_mod: VisionArray, img: VisionArray, rng: VisionArray, shape_info: tuple[int, int, int, int]
) -> VisionArray:
    """Extract a random crop from a single image and pad if necessary.

    Args:
        np_mod: The numpy-like module for array operations.
        img: The single image array to crop.
        rng: The random number generator instance.
        shape_info (tuple[int, int, int, int]): The dimensions information containing crop size and padding.

    Returns: VisionArray: The cropped and padded image array.
    """
    return (0, 0)


def _compute_random_crop(np_mod: VisionArray, imgs: VisionArray, config: RandomCropConfig | None = None) -> VisionArray:
    """Apply random cropping across a batch of images.

    Args:
        np_mod: The numpy-like module for array operations.
        imgs: The batch of images to crop.
        config (RandomCropConfig | None, optional): The configuration specifying crop sizes. Defaults to None.

    Returns: VisionArray: The batch of cropped images.
    """
    return (0, 0)


def random_crop_eager(
    backend_module: VisionArray, images: VisionArray, size: tuple[int, int], seed: VisionArray | None = None
) -> VisionArray:
    """Execute a random spatial crop on a batch of images eagerly.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        images: Input `images` argument (images).
        size (tuple[int, int]): Input `size` argument (size).
        seed: Input `seed` argument (seed).

    Returns:
            VisionArray: Computed `VisionArray` output.
    """
    return (0, 0)


def random_perspective_eager(
    backend_module: VisionArray, images: VisionArray, factor: float | tuple[float, float], **kwargs: VisionArray
) -> VisionArray:
    """Apply a random perspective transformation to a batch of images eagerly.

    Args:
        backend_module: The backend module to use for array operations.
        images: The input tensor of images to transform.
        factor (float | tuple[float, float]): The severity factor of the perspective distortion.
        **kwargs: Additional transformation configuration options.

    Returns: VisionArray: The batch of distorted images.
    """
    return (0, 0)


def _blur_displacement(np_mod: VisionArray, d: VisionArray, s: float) -> VisionArray:
    """Smooth a displacement field using a Gaussian-like blur.

    Args:
        np_mod: The numpy-like module for array operations.
        d: The raw displacement field array.
        s (float): The sigma value controlling the blur spread.

    Returns: VisionArray: The smoothed displacement field.
    """
    return (0, 0)


def _generate_random_elastic_grid(
    np_mod: VisionArray, shape: tuple[int, int, int], rng: VisionArray, a: float, s: float
) -> tuple[VisionArray, VisionArray]:
    """Create a displacement grid for elastic transformations using smoothed random noise.

    Args:
        np_mod: The numpy-like module for array operations.
        shape (tuple[int, int, int]): The shape of the grid to generate (B, H, W).
        rng: The random number generator instance.
        a (float): The alpha scaling factor for the displacement magnitude.
        s (float): The sigma value for the Gaussian smoothing.

    Returns:
        tuple[VisionArray, VisionArray]: The Y and X displacement grids.
    """
    return (0, 0)


def _get_elastic_factor(rng: VisionArray, f: VisionArray) -> float:
    """Sample an elastic parameter from a uniform distribution or return a constant.

    Args:
        rng: The random number generator instance.
        f: A scalar factor or a tuple representing a uniform range.

    Returns:
        float: The sampled configuration parameter for elastic distortion.
    """
    if isinstance(f, (tuple, list)):
        return coerce_float(adapt_backend_module(rng).require_op("uniform")(f[0], f[1]))
    return float(coerce_float(f))


def random_elastic_transform_eager(
    backend_module: types.ModuleType | BackendModuleProtocol | None,
    images: VisionArray,
    alpha: float | tuple[float, float],
    sigma: float | tuple[float, float],
    **kwargs: VisionArray,
) -> VisionArray:
    """Apply random elastic distortion to a batch of images eagerly.

    Args:
        backend_module: The backend module to use for array operations.
        images: The input tensor of images to distort.
        alpha (float | tuple[float, float]): The magnitude scaling factor for displacements.
        sigma (float | tuple[float, float]): The smoothness factor for the displacement field.
        **kwargs: Additional configuration parameters like interpolation or padding mode.

    Returns: VisionArray: The batch of elastically transformed images.
    """
    from .vision_utils import _resolve_backend_mod

    data_format = kwargs.get("data_format", None) if adapt_backend_module(kwargs).has_symbol("get") else None
    ctx = _prepare_eager_transform(
        backend_module,
        images,
        kwargs.get("seed", None) if adapt_backend_module(kwargs).has_symbol("get") else None,
        data_format,
    )
    np_mod = _resolve_backend_mod(ctx.np_mod)
    a = _get_elastic_factor(ctx.rng, alpha)
    s = _get_elastic_factor(ctx.rng, sigma)
    (new_y, new_x) = _generate_random_elastic_grid(np_mod, (ctx.B, ctx.H, ctx.W), ctx.rng, a, s)
    t_config = TransformInterpolationConfig(
        new_y=new_y,
        new_x=new_x,
        order=1
        if str(
            kwargs.get("interpolation", "bilinear") if adapt_backend_module(kwargs).has_symbol("get") else "bilinear"
        )
        == "bilinear"
        else 0,
        fill_value=float(
            coerce_float(kwargs.get("fill_value", 0.0) if adapt_backend_module(kwargs).has_symbol("get") else 0.0)
        ),
    )
    out = _apply_elastic_batch(np_mod, ctx.imgs, t_config)
    out_cl = _from_channels_last(np_mod, out, coerce_opt_str(data_format))
    return _from_numpy_array(backend_module, out_cl, "", images)


def _compute_zoom_grid(
    np_mod: types.ModuleType | BackendModuleProtocol, config: GeometricGridConfig
) -> tuple[VisionArray, VisionArray]:
    """Calculate the destination spatial grid for a zoom transformation.

    Args:
        np_mod: The numpy-like module for array operations.
        config (GeometricGridConfig): The configuration containing zoom factors and image dimensions.

    Returns:
        tuple[VisionArray, VisionArray]: The Y and X coordinates of the zoomed grid.
    """
    ys, xs = _generate_coordinate_grid(np_mod, config.H, config.W)
    sy = 1.0 + _get_shear_factor(config.rng, config.factor1)
    sx = 1.0 + _get_shear_factor(config.rng, config.factor2 if config.factor2 is not None else config.factor1)
    cy = (config.H - 1) / 2.0
    cx = (config.W - 1) / 2.0
    return ((ys - cy) * sy + cy, (xs - cx) * sx + cx)


def _compute_translation_grid(
    np_mod: types.ModuleType | BackendModuleProtocol, config: GeometricGridConfig
) -> tuple[VisionArray, VisionArray]:
    """Calculate the shifted spatial grid for a translation operation.

    Args:
        np_mod: The numpy-like module for array operations.
        config (GeometricGridConfig): The configuration containing translation offsets and grid size.

    Returns:
        tuple[VisionArray, VisionArray]: The Y and X coordinates of the translated grid.
    """
    ys, xs = _generate_coordinate_grid(np_mod, config.H, config.W)
    dy = _get_shear_factor(config.rng, config.factor1) * float(config.H)
    dx = _get_shear_factor(config.rng, config.factor2) * float(config.W)
    return (ys - dy, xs - dx)


def _get_shear_factor(rng: VisionArray, factor: VisionArray) -> float:
    """Sample a shear amount from a symmetric range or a specified interval.

    Args:
        rng: The random number generator instance.
        factor: A scalar defining the symmetric range or a tuple defining min/max shear.

    Returns:
        float: The sampled shear magnitude.
    """
    if isinstance(factor, (tuple, list)):
        return coerce_float(adapt_backend_module(rng).require_op("uniform")(factor[0], factor[1]))
    f_val = coerce_float(factor)
    return coerce_float(adapt_backend_module(rng).require_op("uniform")(-f_val, f_val))


def _compute_shear_grid(
    np_mod: types.ModuleType | BackendModuleProtocol, config: GeometricGridConfig
) -> tuple[VisionArray, VisionArray]:
    """Calculate the skewed spatial grid for a shear transformation.

    Args:
        np_mod: The numpy-like module for array operations.
        config (GeometricGridConfig): The configuration containing shear factors and image dimensions.

    Returns:
        tuple[VisionArray, VisionArray]: The Y and X coordinates of the sheared grid.
    """
    ys, xs = _generate_coordinate_grid(np_mod, config.H, config.W)
    shy = _get_shear_factor(config.rng, config.factor1)
    shx = _get_shear_factor(config.rng, config.factor2 if config.factor2 is not None else 0.0)
    return (ys + shy * xs, xs + shx * ys)


def random_zoom_eager(
    backend_module: types.ModuleType | BackendModuleProtocol | None,
    images: VisionArray,
    height_factor: tuple[float, float] | float,
    width_factor: tuple[float, float] | float | None = None,
    **kwargs: VisionArray,
) -> VisionArray:
    """Apply a random zoom operation to a batch of images eagerly.

    Args:
        backend_module: The backend module to use for array operations.
        images: The input tensor of images to zoom.
        height_factor (tuple[float, float] | float): The zoom scaling factor range for the vertical axis.
        width_factor (tuple[float, float] | float | None, optional): The zoom scaling factor range for the horizontal
            axis. Defaults to None.
        **kwargs: Additional configuration like interpolation or fill mode.

    Returns: VisionArray: The batch of zoomed images.
    """
    from .vision_utils import _resolve_backend_mod

    fill_mode = str(kwargs.get("fill_mode", "reflect") if adapt_backend_module(kwargs).has_symbol("get") else "reflect")
    interpolation = str(
        kwargs.get("interpolation", "bilinear") if adapt_backend_module(kwargs).has_symbol("get") else "bilinear"
    )
    fill_value = float(
        coerce_float(kwargs.get("fill_value", 0.0) if adapt_backend_module(kwargs).has_symbol("get") else 0.0)
    )
    seed = kwargs.get("seed", None) if adapt_backend_module(kwargs).has_symbol("get") else None
    data_format = kwargs.get("data_format", None) if adapt_backend_module(kwargs).has_symbol("get") else None
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)
    np_mod = _resolve_backend_mod(ctx.np_mod)
    if width_factor is None:
        width_factor = height_factor
    (new_y, new_x) = _compute_zoom_grid(
        np_mod, GeometricGridConfig(H=ctx.H, W=ctx.W, rng=ctx.rng, factor1=height_factor, factor2=width_factor)
    )
    config = RotationConfig(
        factor=0.0,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=coerce_str(data_format),
    )
    out_arr = _interpolate_pixels(np_mod, ctx.imgs, new_y, new_x, config)
    out = coerce_array(np_mod, _from_channels_last(np_mod, out_arr, coerce_opt_str(data_format)))
    out_res: VisionArray = out[0] if adapt_backend_module(images).get_symbol("ndim", 4) == 3 else out
    return _from_numpy_array(backend_module, out_res, "", images)


def random_translation_eager(
    backend_module: types.ModuleType | BackendModuleProtocol | None,
    images: VisionArray,
    height_factor: tuple[float, float] | float,
    width_factor: tuple[float, float] | float,
    **kwargs: VisionArray,
) -> VisionArray:
    """Apply a random spatial translation to a batch of images eagerly.

    Args:
        backend_module: The backend module to use for array operations.
        images: The input tensor of images to shift.
        height_factor (tuple[float, float] | float): The translation fraction range for the vertical axis.
        width_factor (tuple[float, float] | float): The translation fraction range for the horizontal axis.
        **kwargs: Additional configuration parameters like interpolation mode.

    Returns: VisionArray: The batch of translated images.
    """
    from .vision_utils import _resolve_backend_mod

    fill_mode = str(kwargs.get("fill_mode", "reflect") if adapt_backend_module(kwargs).has_symbol("get") else "reflect")
    interpolation = str(
        kwargs.get("interpolation", "bilinear") if adapt_backend_module(kwargs).has_symbol("get") else "bilinear"
    )
    fill_value = float(
        coerce_float(kwargs.get("fill_value", 0.0) if adapt_backend_module(kwargs).has_symbol("get") else 0.0)
    )
    seed = kwargs.get("seed", None) if adapt_backend_module(kwargs).has_symbol("get") else None
    data_format = kwargs.get("data_format", None) if adapt_backend_module(kwargs).has_symbol("get") else None
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)
    np_mod = _resolve_backend_mod(ctx.np_mod)
    (new_y, new_x) = _compute_translation_grid(
        np_mod, GeometricGridConfig(H=ctx.H, W=ctx.W, rng=ctx.rng, factor1=height_factor, factor2=width_factor)
    )
    config = RotationConfig(
        factor=0.0,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=coerce_str(data_format),
    )
    out_arr = _interpolate_pixels(np_mod, ctx.imgs, new_y, new_x, config)
    out = coerce_array(np_mod, _from_channels_last(np_mod, out_arr, coerce_opt_str(data_format)))
    out_res: VisionArray = out[0] if adapt_backend_module(images).get_symbol("ndim", 4) == 3 else out
    return _from_numpy_array(backend_module, out_res, "", images)


def random_shear_eager(
    backend_module: types.ModuleType | BackendModuleProtocol | None,
    images: VisionArray,
    y_factor: tuple[float, float] | float,
    x_factor: tuple[float, float] | float | None = None,
    **kwargs: VisionArray,
) -> VisionArray:
    """Apply a random affine shear to a batch of images eagerly.

    Args:
        backend_module: The backend module to use for array operations.
        images: The input tensor of images to shear.
        y_factor (tuple[float, float] | float): The shearing factor magnitude for the vertical axis.
        x_factor (tuple[float, float] | float | None, optional): The shearing factor magnitude for the horizontal
            axis. Defaults to None.
        **kwargs: Additional configuration variables, e.g., interpolation.

    Returns: VisionArray: The batch of sheared images.
    """
    from .vision_utils import _resolve_backend_mod

    fill_mode = str(kwargs.get("fill_mode", "reflect") if adapt_backend_module(kwargs).has_symbol("get") else "reflect")
    interpolation = str(
        kwargs.get("interpolation", "bilinear") if adapt_backend_module(kwargs).has_symbol("get") else "bilinear"
    )
    fill_value = float(
        coerce_float(kwargs.get("fill_value", 0.0) if adapt_backend_module(kwargs).has_symbol("get") else 0.0)
    )
    seed = kwargs.get("seed", None) if adapt_backend_module(kwargs).has_symbol("get") else None
    data_format = kwargs.get("data_format", None) if adapt_backend_module(kwargs).has_symbol("get") else None
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)
    np_mod = _resolve_backend_mod(ctx.np_mod)
    (new_y, new_x) = _compute_shear_grid(
        np_mod, GeometricGridConfig(H=ctx.H, W=ctx.W, rng=ctx.rng, factor1=y_factor, factor2=x_factor)
    )
    config = RotationConfig(
        factor=0.0,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=coerce_str(data_format),
    )
    out_arr = _interpolate_pixels(np_mod, ctx.imgs, new_y, new_x, config)
    out_cl = _from_channels_last(np_mod, out_arr, coerce_opt_str(data_format))
    return _from_numpy_array(backend_module, out_cl, "", images)
