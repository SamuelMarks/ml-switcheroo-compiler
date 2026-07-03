"""Vision utilities."""

from __future__ import annotations

from dataclasses import dataclass

from ml_switcheroo_compiler.backends.eager.signal import _np_gaussian_blur
from ml_switcheroo_compiler.backends.eager.utils import (
    _from_channels_last,
    _from_numpy_array,
    _to_channels_last,
    _to_numpy_array,
)
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_0_5
from ml_switcheroo_compiler.ops.configs import PerspectiveConfig

from .vision_transforms import ElasticGridContext, _apply_elastic_batch, _compute_elastic_grid
from .vision_utils import (
    GeometricGridConfig,
    RandomCropConfig,
    TransformInterpolationConfig,
    _apply_perspective_batch,
    _compute_perspective_matrix,
    _np_map_coordinates,
    _prepare_eager_transform,
)


def random_flip_eager(backend_module: object, images: object, mode: str, seed: object = None) -> object:
    """Evaluate random flip eagerly."""
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(seed)
    if mode in ("horizontal", "horizontal_and_vertical"):
        if rng.random() > MAGIC_VAL_0_5:
            imgs = np_mod.flip(imgs, axis=-2)
    if mode in ("vertical", "horizontal_and_vertical"):
        if rng.random() > MAGIC_VAL_0_5:
            imgs = np_mod.flip(imgs, axis=-3)
    return _from_numpy_array(backend_module, imgs, name, images)


@dataclass
class RotationConfig:
    """Configuration for random rotation."""

    factor: float
    fill_mode: str
    interpolation: str
    seed: object
    fill_value: float
    data_format: str


def _compute_rotation_matrix(np_mod: object, angle: float, W: int, H: int) -> tuple[float, float, float, float]:
    """Compute 2D affine rotation matrix params."""
    return (np_mod.cos(angle), np_mod.sin(angle), W / 2.0, H / 2.0)


def _generate_coordinate_grid(np_mod: object, H: int, W: int) -> tuple[object, object]:
    """Generate 2D meshgrid coordinates."""
    return np_mod.meshgrid(np_mod.arange(H), np_mod.arange(W), indexing="ij")


@dataclass
class AffineTransformParams:
    """Class docstring."""

    cos_a: float
    sin_a: float
    cx: float
    cy: float


def _apply_affine_transform(y_grid: object, x_grid: object, params: AffineTransformParams) -> tuple[object, object]:
    """Apply affine transformation to coordinates."""
    x_shifted = x_grid - params.cx
    y_shifted = y_grid - params.cy
    x_rot = x_shifted * params.cos_a + y_shifted * params.sin_a
    y_rot = -x_shifted * params.sin_a + y_shifted * params.cos_a
    return (y_rot + params.cy, x_rot + params.cx)


def _interpolate_pixels(np_mod: object, imgs: object, new_y: object, new_x: object, config: RotationConfig) -> object:
    """Interpolate pixels."""
    (B, H, W, C) = imgs.shape
    out = np_mod.zeros_like(imgs)
    order = 1 if config.interpolation == "bilinear" else 0
    for b in range(B):
        for c in range(C):
            out[b, ..., c] = _np_map_coordinates(np_mod, imgs[b, ..., c], [new_y, new_x], order=order, fill_value=config.fill_value)
    return out


def _compute_rotation_grid(np_mod: object, H: int, W: int, rng: object, factor: float) -> tuple[object, object]:
    """Function docstring.

    Args:
        np_mod: Arg.
        H: Arg.
        W: Arg.
        rng: Arg.
        factor: Arg.
    """
    angle_rad = rng.uniform(-factor, factor) * np_mod.pi / 180.0
    cos_a = np_mod.cos(angle_rad)
    sin_a = np_mod.sin(angle_rad)
    (y_grid, x_grid) = _generate_coordinate_grid(np_mod, H, W)
    (cy, cx) = (H / 2.0, W / 2.0)
    return _apply_affine_transform(y_grid, x_grid, AffineTransformParams(cos_a, sin_a, cx, cy))


def random_rotation_eager(backend_module: object, images: object, config: RotationConfig) -> object:
    """Evaluate random rotation eagerly."""
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(config.seed)
    angle = rng.uniform(-config.factor * 2 * np_mod.pi, config.factor * 2 * np_mod.pi)
    imgs = _to_channels_last(np_mod, imgs, config.data_format)
    (B, H, W, C) = imgs.shape
    (cos_a, sin_a, cx, cy) = _compute_rotation_matrix(np_mod, angle, W, H)
    (y_grid, x_grid) = _generate_coordinate_grid(np_mod, H, W)
    (new_y, new_x) = _apply_affine_transform(y_grid, x_grid, AffineTransformParams(cos_a, sin_a, cx, cy))
    out = _interpolate_pixels(np_mod, imgs, new_y, new_x, config)
    out = _from_channels_last(np_mod, out, config.data_format)
    return _from_numpy_array(backend_module, out, name, images)


def _crop_and_pad_single(np_mod: object, img: object, rng: object, shape_info: tuple[int, int, int, int]) -> object:
    """Function docstring."""
    (H, W, height, width) = shape_info
    y_start = rng.integers(0, H - height + 1) if H >= height else 0
    x_start = rng.integers(0, W - width + 1) if W >= width else 0
    y_end = min(y_start + height, H)
    x_end = min(x_start + width, W)
    cropped = img[y_start:y_end, x_start:x_end, :]
    pad_y = height - cropped.shape[0]
    pad_x = width - cropped.shape[1]
    if pad_y > 0 or pad_x > 0:
        cropped = np_mod.pad(cropped, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant")
    return cropped


def _compute_random_crop(np_mod: object, imgs: object, config: RandomCropConfig | None = None) -> object:
    """Function docstring."""
    conf = config if config is not None else RandomCropConfig(0, 0, 0, 0, 0, 0, None)
    out = np_mod.zeros((conf.b, conf.crop_h, conf.crop_w, conf.c), dtype=imgs.dtype)
    for b in range(conf.b):
        out[b] = _crop_and_pad_single(np_mod, imgs[b], conf.rng, conf.H, conf.W, conf.crop_h, conf.crop_w)
    return out


def random_crop_eager(backend_module: object, images: object, size: tuple, seed: object = None) -> object:
    """Evaluate random crop eagerly."""
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(seed)
    (B, H, W, C) = imgs.shape
    (new_H, new_W) = size
    if H <= new_H and W <= new_W:
        return images
    start_h = rng.integers(0, H - new_H + 1) if H > new_H else 0
    start_w = rng.integers(0, W - new_W + 1) if W > new_W else 0
    out = imgs[:, start_h : start_h + new_H, start_w : start_w + new_W, :]
    return _from_numpy_array(backend_module, out, name, images)


def random_perspective_eager(backend_module: object, images: object, factor: float | tuple[float, float], **kwargs: object) -> object:
    """Evaluate random perspective eagerly."""
    seed = kwargs.get("seed", None)
    data_format = kwargs.get("data_format", None)
    interpolation = str(kwargs.get("interpolation", "bilinear"))
    fill_value = float(kwargs.get("fill_value", 0.0))
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)
    np_mod = ctx.np_mod
    (B, H, W) = (ctx.B, ctx.H, ctx.W)

    def get_factor(f: object) -> float:
        """Function docstring.

        Args:
            f: Arg.
        """
        if isinstance(f, (tuple, list)):
            return ctx.rng.uniform(f[0], f[1])
        return ctx.rng.uniform(0, f)

    dist = get_factor(factor)
    src = np_mod.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np_mod.float32)
    src = np_mod.broadcast_to(src, (B, 4, 2))
    dx = ctx.rng.uniform(-dist * W, dist * W, size=(B, 4, 1))
    dy = ctx.rng.uniform(-dist * H, dist * H, size=(B, 4, 1))
    jitter = np_mod.concatenate([dx, dy], axis=-1)
    dst = src + jitter
    h = _compute_perspective_matrix(np_mod, src, dst)
    p_config = PerspectiveConfig(interpolation=interpolation, fill_value=fill_value, data_format=data_format)
    out = _apply_perspective_batch(np_mod, ctx.imgs, h, p_config)
    out = _from_channels_last(ctx.np_mod, out, data_format)
    return _from_numpy_array(backend_module, out, "", images)


def _blur_displacement(np_mod: object, d: object, s: float) -> object:
    """Function docstring."""
    return _np_gaussian_blur(np_mod, d[..., None], (int(s * 4 + 1), int(s * 4 + 1)), (s, s))[..., 0]


def _generate_random_elastic_grid(np_mod: object, shape: tuple[int, int, int], rng: object, a: float, s: float) -> tuple[object, object]:
    """Generate elastic transformation grid."""
    dx = rng.uniform(-1, 1, size=shape)
    dy = rng.uniform(-1, 1, size=shape)
    dx_disp = _blur_displacement(np_mod, dx, s) * a
    dy_disp = _blur_displacement(np_mod, dy, s) * a
    disp = np_mod.stack([dy_disp, dx_disp], axis=-1)
    ctx = ElasticGridContext(np_mod, shape[1], shape[2], shape[0], disp)
    return _compute_elastic_grid(ctx)


def _get_elastic_factor(rng: object, f: object) -> float:
    """Function docstring."""
    if isinstance(f, (tuple, list)):
        return rng.uniform(f[0], f[1])
    return float(f)


def random_elastic_transform_eager(
    backend_module: object,
    images: object,
    alpha: float | tuple[float, float],
    sigma: float | tuple[float, float],
    **kwargs: object,
) -> object:
    """Evaluate random elastic transform eagerly."""
    data_format = kwargs.get("data_format", None)
    ctx = _prepare_eager_transform(backend_module, images, kwargs.get("seed", None), data_format)
    a = _get_elastic_factor(ctx.rng, alpha)
    s = _get_elastic_factor(ctx.rng, sigma)
    (new_y, new_x) = _generate_random_elastic_grid(ctx.np_mod, (ctx.B, ctx.H, ctx.W), ctx.rng, a, s)
    t_config = TransformInterpolationConfig(
        new_y=new_y,
        new_x=new_x,
        order=1 if str(kwargs.get("interpolation", "bilinear")) == "bilinear" else 0,
        fill_value=float(kwargs.get("fill_value", 0.0)),
    )
    out = _apply_elastic_batch(ctx.np_mod, ctx.imgs, t_config)
    out = _from_channels_last(ctx.np_mod, out, data_format)
    return _from_numpy_array(backend_module, out, "", images)


def _compute_zoom_grid(np_mod: object, config: GeometricGridConfig) -> tuple[object, object]:
    """Function docstring."""
    H = config.H
    W = config.W
    rng = config.rng
    height_factor = config.factor1
    width_factor = config.factor2
    "Function docstring.\n\n    Args:\n        np_mod: Arg.\n        H: Arg.\n        W: Arg.\n        rng: Arg.\n        height_factor: Arg.\n        width_factor: Arg.\n    "

    def get_factor(factor: object) -> float:
        """Function docstring.

        Args:
        factor: Arg.
        """
        if isinstance(factor, (tuple, list)):
            return rng.uniform(factor[0], factor[1])
        return rng.uniform(1.0 - factor, 1.0 + factor)

    zx = get_factor(width_factor)
    zy = get_factor(height_factor)
    (y_grid, x_grid) = _generate_coordinate_grid(np_mod, H, W)
    (cy, cx) = (H / 2.0, W / 2.0)
    return ((y_grid - cy) / zy + cy, (x_grid - cx) / zx + cx)


def _compute_translation_grid(np_mod: object, config: GeometricGridConfig) -> tuple[object, object]:
    """Function docstring."""
    H = config.H
    W = config.W
    rng = config.rng
    height_factor = config.factor1
    width_factor = config.factor2
    "Function docstring.\n\n    Args:\n        np_mod: Arg.\n        H: Arg.\n        W: Arg.\n        rng: Arg.\n        height_factor: Arg.\n        width_factor: Arg.\n    "

    def get_factor(factor: object) -> float:
        """Function docstring.

        Args:
        factor: Arg.
        """
        if isinstance(factor, (tuple, list)):
            return rng.uniform(factor[0], factor[1])
        return rng.uniform(-factor, factor)

    tx = get_factor(width_factor) * W
    ty = get_factor(height_factor) * H
    (y_grid, x_grid) = _generate_coordinate_grid(np_mod, H, W)
    return (y_grid - ty, x_grid - tx)


def _get_shear_factor(rng: object, factor: object) -> float:
    """Function docstring."""
    if isinstance(factor, (tuple, list)):
        return rng.uniform(factor[0], factor[1])
    return rng.uniform(-factor, factor)


def _compute_shear_grid(np_mod: object, config: GeometricGridConfig) -> tuple[object, object]:
    """Function docstring."""
    sy = _get_shear_factor(config.rng, config.factor1)
    sx = _get_shear_factor(config.rng, config.factor2) if config.factor2 is not None else 0.0
    (y_grid, x_grid) = _generate_coordinate_grid(np_mod, config.H, config.W)
    (cy, cx) = (config.H / 2.0, config.W / 2.0)
    y_shifted = y_grid - cy
    x_shifted = x_grid - cx
    return (y_shifted - sy * x_shifted + cy, x_shifted - sx * y_shifted + cx)


def random_zoom_eager(
    backend_module: object,
    images: object,
    height_factor: tuple[float, float] | float,
    width_factor: tuple[float, float] | float | None = None,
    **kwargs: object,
) -> object:
    """Evaluate random zoom eagerly."""
    fill_mode = str(kwargs.get("fill_mode", "reflect"))
    interpolation = str(kwargs.get("interpolation", "bilinear"))
    fill_value = float(kwargs.get("fill_value", 0.0))
    seed = kwargs.get("seed", None)
    data_format = kwargs.get("data_format", None)
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)
    if width_factor is None:
        width_factor = height_factor
    (new_y, new_x) = _compute_zoom_grid(ctx.np_mod, ctx.H, ctx.W, ctx.rng, height_factor, width_factor)
    config = RotationConfig(
        factor=0.0,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=data_format,
    )
    out = _interpolate_pixels(ctx.np_mod, ctx.imgs, new_y, new_x, config)
    out = _from_channels_last(ctx.np_mod, out, data_format)
    return _from_numpy_array(backend_module, out, "", images)


def random_translation_eager(
    backend_module: object,
    images: object,
    height_factor: tuple[float, float] | float,
    width_factor: tuple[float, float] | float,
    **kwargs: object,
) -> object:
    """Evaluate random translation eagerly."""
    fill_mode = str(kwargs.get("fill_mode", "reflect"))
    interpolation = str(kwargs.get("interpolation", "bilinear"))
    fill_value = float(kwargs.get("fill_value", 0.0))
    seed = kwargs.get("seed", None)
    data_format = kwargs.get("data_format", None)
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)
    (new_y, new_x) = _compute_translation_grid(ctx.np_mod, ctx.H, ctx.W, ctx.rng, height_factor, width_factor)
    config = RotationConfig(
        factor=0.0,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=data_format,
    )
    out = _interpolate_pixels(ctx.np_mod, ctx.imgs, new_y, new_x, config)
    out = _from_channels_last(ctx.np_mod, out, data_format)
    return _from_numpy_array(backend_module, out, "", images)


def random_shear_eager(
    backend_module: object,
    images: object,
    y_factor: tuple[float, float] | float,
    x_factor: tuple[float, float] | float | None = None,
    **kwargs: object,
) -> object:
    """Evaluate random shear eagerly."""
    fill_mode = str(kwargs.get("fill_mode", "reflect"))
    interpolation = str(kwargs.get("interpolation", "bilinear"))
    fill_value = float(kwargs.get("fill_value", 0.0))
    seed = kwargs.get("seed", None)
    data_format = kwargs.get("data_format", None)
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)
    (new_y, new_x) = _compute_shear_grid(ctx.np_mod, ctx.H, ctx.W, ctx.rng, y_factor, x_factor)
    config = RotationConfig(
        factor=0.0,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=data_format,
    )
    out = _interpolate_pixels(ctx.np_mod, ctx.imgs, new_y, new_x, config)
    out = _from_channels_last(ctx.np_mod, out, data_format)
    return _from_numpy_array(backend_module, out, "", images)
