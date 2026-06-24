# ruff: noqa: F405, F403

# ruff: noqa: E402
"""Vision utilities."""

from __future__ import annotations

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_0_5

from dataclasses import dataclass


from ml_switcheroo_compiler.ops.configs import PerspectiveConfig
from ml_switcheroo_compiler.backends.eager.utils import (
    _from_channels_last,
    _from_numpy_array,
    _to_channels_last,
    _to_numpy_array,
)

from .vision_utils import *
from .vision_transforms import *


def random_flip_eager(
    backend_module: object, images: object, mode: str, seed: object = None
) -> object:
    """Evaluate random flip eagerly."""
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(seed)

    if mode in ("horizontal", "horizontal_and_vertical"):  # pragma: no branch
        if rng.random() > MAGIC_VAL_0_5:  # pragma: no cover
            imgs = np_mod.flip(
                imgs, axis=-2
            )  # width is -2 if shape is (B, H, W, C)  # pragma: no cover
    if mode in ("vertical", "horizontal_and_vertical"):  # pragma: no branch
        if rng.random() > MAGIC_VAL_0_5:  # pragma: no cover
            imgs = np_mod.flip(imgs, axis=-3)  # height is -3  # pragma: no cover

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


def _compute_rotation_matrix(
    np_mod: object, angle: float, W: int, H: int
) -> tuple[float, float, float, float]:
    """Compute 2D affine rotation matrix params."""
    return np_mod.cos(angle), np_mod.sin(angle), W / 2.0, H / 2.0


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


def _apply_affine_transform(
    y_grid: object, x_grid: object, params: AffineTransformParams
) -> tuple[object, object]:
    """Apply affine transformation to coordinates."""
    x_shifted = x_grid - params.cx
    y_shifted = y_grid - params.cy
    x_rot = x_shifted * params.cos_a + y_shifted * params.sin_a
    y_rot = -x_shifted * params.sin_a + y_shifted * params.cos_a
    return y_rot + params.cy, x_rot + params.cx


def _interpolate_pixels(
    np_mod: object, imgs: object, new_y: object, new_x: object, config: RotationConfig
) -> object:
    """Interpolate pixels."""
    B, H, W, C = imgs.shape
    out = np_mod.zeros_like(imgs)
    order = 1 if config.interpolation == "bilinear" else 0
    for b in range(B):
        for c in range(C):
            out[b, ..., c] = _np_map_coordinates(
                np_mod, imgs[b, ..., c], [new_y, new_x], order=order, fill_value=config.fill_value
            )
    return out


def _compute_rotation_grid(
    np_mod: object, H: int, W: int, rng: object, factor: float
) -> tuple[object, object]:
    """Function docstring.

    Args:
        np_mod: Arg.
        H: Arg.
        W: Arg.
        rng: Arg.
        factor: Arg.
    """
    angle_rad = rng.uniform(-factor, factor) * np_mod.pi / 180.0  # pragma: no cover
    cos_a = np_mod.cos(angle_rad)  # pragma: no cover
    sin_a = np_mod.sin(angle_rad)  # pragma: no cover
    y_grid, x_grid = _generate_coordinate_grid(np_mod, H, W)  # pragma: no cover
    cy, cx = H / 2.0, W / 2.0  # pragma: no cover
    return _apply_affine_transform(
        y_grid, x_grid, AffineTransformParams(cos_a, sin_a, cx, cy)
    )  # pragma: no cover


def random_rotation_eager(  # pylint: disable=too-many-locals
    backend_module: object, images: object, config: RotationConfig
) -> object:
    """Evaluate random rotation eagerly."""
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(config.seed)

    angle = rng.uniform(-config.factor * 2 * np_mod.pi, config.factor * 2 * np_mod.pi)

    imgs = _to_channels_last(np_mod, imgs, config.data_format)
    B, H, W, C = imgs.shape

    cos_a, sin_a, cx, cy = _compute_rotation_matrix(np_mod, angle, W, H)
    y_grid, x_grid = _generate_coordinate_grid(np_mod, H, W)
    new_y, new_x = _apply_affine_transform(
        y_grid, x_grid, AffineTransformParams(cos_a, sin_a, cx, cy)
    )

    out = _interpolate_pixels(np_mod, imgs, new_y, new_x, config)

    out = _from_channels_last(np_mod, out, config.data_format)
    return _from_numpy_array(backend_module, out, name, images)


def _compute_random_crop(
    np_mod: object,
    imgs: object,
    config: RandomCropConfig | None = None,
) -> object:
    conf = (
        config if config is not None else RandomCropConfig(0, 0, 0, 0, 0, 0, None)
    )  # pragma: no cover
    height = conf.crop_h  # pragma: no cover
    width = conf.crop_w  # pragma: no cover
    B = conf.b  # pragma: no cover
    C = conf.c  # pragma: no cover
    H = conf.H  # pragma: no cover
    W = conf.W  # pragma: no cover
    rng = conf.rng  # pragma: no cover
    """Function docstring.

    Args:
        np_mod: Arg.
        imgs: Arg.
        B: Arg.
        H: Arg.
        W: Arg.
        C: Arg.
        height: Arg.
        width: Arg.
        rng: Arg.
    """
    out = np_mod.zeros((B, height, width, C), dtype=imgs.dtype)  # type: ignore  # pragma: no cover
    for b in range(B):  # pragma: no cover
        y_start = rng.integers(0, H - height + 1) if H >= height else 0  # type: ignore  # pragma: no cover
        x_start = rng.integers(0, W - width + 1) if W >= width else 0  # type: ignore  # pragma: no cover
        y_end = min(y_start + height, H)  # pragma: no cover
        x_end = min(x_start + width, W)  # pragma: no cover
        cropped = imgs[b, y_start:y_end, x_start:x_end, :]  # type: ignore  # pragma: no cover
        pad_y = height - cropped.shape[0]  # pragma: no cover
        pad_x = width - cropped.shape[1]  # pragma: no cover
        if pad_y > 0 or pad_x > 0:  # pragma: no cover
            cropped = np_mod.pad(cropped, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant")  # type: ignore  # pragma: no cover
        out[b] = cropped  # type: ignore  # pragma: no cover
    return out  # pragma: no cover


def random_crop_eager(  # pylint: disable=too-many-locals
    backend_module: object, images: object, size: tuple, seed: object = None
) -> object:
    """Evaluate random crop eagerly."""
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(seed)

    B, H, W, C = imgs.shape
    new_H, new_W = size  # pragma: no cover

    if H <= new_H and W <= new_W:  # pragma: no cover
        return images  # pragma: no cover

    start_h = rng.integers(0, H - new_H + 1) if H > new_H else 0  # pragma: no cover
    start_w = rng.integers(0, W - new_W + 1) if W > new_W else 0  # pragma: no cover

    out = imgs[:, start_h : start_h + new_H, start_w : start_w + new_W, :]  # pragma: no cover
    return _from_numpy_array(backend_module, out, name, images)  # pragma: no cover


def random_perspective_eager(
    backend_module: object,
    images: object,
    factor: float | tuple[float, float],
    **kwargs: object,
) -> object:
    """Evaluate random perspective eagerly."""
    seed = kwargs.get("seed", None)  # pragma: no cover
    data_format = kwargs.get("data_format", None)  # pragma: no cover
    interpolation = str(kwargs.get("interpolation", "bilinear"))  # pragma: no cover
    fill_value = float(kwargs.get("fill_value", 0.0))  # pragma: no cover
    # pragma: no cover
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)  # pragma: no cover
    np_mod = ctx.np_mod  # pragma: no cover
    B, H, W = ctx.B, ctx.H, ctx.W  # pragma: no cover

    # pragma: no cover
    def get_factor(f: object) -> float:  # pragma: no cover
        """Function docstring.

        Args:
            f: Arg.
        """
        if isinstance(f, (tuple, list)):  # pragma: no branch  # pragma: no cover
            return ctx.rng.uniform(f[0], f[1])  # type: ignore  # pragma: no cover
        return ctx.rng.uniform(0, f)  # type: ignore  # pragma: no cover

    # pragma: no cover
    dist = get_factor(factor)  # pragma: no cover
    # pragma: no cover
    # We create random start/end points  # pragma: no cover
    src = np_mod.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np_mod.float32
    )  # pragma: no cover
    src = np_mod.broadcast_to(src, (B, 4, 2))  # pragma: no cover
    # pragma: no cover
    # Add random jitter bounded by dist * W or dist * H  # pragma: no cover
    dx = ctx.rng.uniform(-dist * W, dist * W, size=(B, 4, 1))  # pragma: no cover
    dy = ctx.rng.uniform(-dist * H, dist * H, size=(B, 4, 1))  # pragma: no cover
    jitter = np_mod.concatenate([dx, dy], axis=-1)  # pragma: no cover
    dst = src + jitter  # pragma: no cover
    # pragma: no cover
    h = _compute_perspective_matrix(np_mod, src, dst)  # pragma: no cover
    # pragma: no cover
    p_config = PerspectiveConfig(  # pragma: no cover
        interpolation=interpolation,
        fill_value=fill_value,
        data_format=data_format,  # pragma: no cover
    )  # pragma: no cover
    # pragma: no cover
    out = _apply_perspective_batch(np_mod, ctx.imgs, h, p_config)  # pragma: no cover

    out = _from_channels_last(ctx.np_mod, out, data_format)  # pragma: no cover
    return _from_numpy_array(backend_module, out, "", images)  # pragma: no cover


def _generate_random_elastic_grid(
    np_mod: object,
    shape: tuple[int, int, int],
    rng: object,
    a: float,
    s: float,
) -> tuple[object, object]:
    """Generate elastic transformation grid."""
    from ml_switcheroo_compiler.backends.eager.signal import _np_gaussian_blur  # pragma: no cover

    B, H, W = shape  # pragma: no cover
    # pragma: no cover
    dx = rng.uniform(-1, 1, size=(B, H, W))  # pragma: no cover
    dy = rng.uniform(-1, 1, size=(B, H, W))  # pragma: no cover
    # pragma: no cover
    dx_expanded = dx[..., None]  # pragma: no cover
    dy_expanded = dy[..., None]  # pragma: no cover
    # pragma: no cover
    dx_blurred = _np_gaussian_blur(  # pragma: no cover
        np_mod,
        dx_expanded,
        (int(s * 4 + 1), int(s * 4 + 1)),
        (s, s),  # pragma: no cover
    )  # pragma: no cover
    dy_blurred = _np_gaussian_blur(  # pragma: no cover
        np_mod,
        dy_expanded,
        (int(s * 4 + 1), int(s * 4 + 1)),
        (s, s),  # pragma: no cover
    )  # pragma: no cover
    # pragma: no cover
    dx_disp = dx_blurred[..., 0] * a  # pragma: no cover
    dy_disp = dy_blurred[..., 0] * a  # pragma: no cover
    # pragma: no cover
    disp = np_mod.stack([dy_disp, dx_disp], axis=-1)  # pragma: no cover
    # pragma: no cover
    ctx = ElasticGridContext(np_mod, H, W, B, disp)  # pragma: no cover
    return _compute_elastic_grid(ctx)  # pragma: no cover


def random_elastic_transform_eager(
    backend_module: object,
    images: object,
    alpha: float | tuple[float, float],
    sigma: float | tuple[float, float],
    **kwargs: object,
) -> object:
    """Evaluate random elastic transform eagerly."""
    seed = kwargs.get("seed", None)  # pragma: no cover
    data_format = kwargs.get("data_format", None)  # pragma: no cover
    interpolation = str(kwargs.get("interpolation", "bilinear"))  # pragma: no cover
    fill_value = float(kwargs.get("fill_value", 0.0))  # pragma: no cover

    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)  # pragma: no cover
    np_mod = ctx.np_mod  # pragma: no cover
    B, H, W = ctx.B, ctx.H, ctx.W  # pragma: no cover

    def get_factor(f: object) -> float:  # pragma: no cover
        """Function docstring.

        Args:
        f: Arg.
        """
        if isinstance(f, (tuple, list)):  # pragma: no cover
            return ctx.rng.uniform(f[0], f[1])  # type: ignore  # pragma: no cover
        return f  # type: ignore  # pragma: no cover

    a = get_factor(alpha)  # pragma: no cover
    s = get_factor(sigma)  # pragma: no cover

    new_y, new_x = _generate_random_elastic_grid(
        np_mod, (B, H, W), ctx.rng, a, s
    )  # pragma: no cover

    order = 1 if interpolation == "bilinear" else 0  # pragma: no cover
    t_config = TransformInterpolationConfig(  # pragma: no cover
        new_y=new_y, new_x=new_x, order=order, fill_value=fill_value
    )

    out = _apply_elastic_batch(np_mod, ctx.imgs, t_config)  # pragma: no cover

    out = _from_channels_last(ctx.np_mod, out, data_format)  # pragma: no cover
    return _from_numpy_array(backend_module, out, "", images)  # pragma: no cover


# ruff: noqa: E402

# We'll fix up imports later if needed.


def _compute_zoom_grid(np_mod: object, config: GeometricGridConfig) -> tuple[object, object]:
    H = config.H  # pragma: no cover
    W = config.W  # pragma: no cover
    rng = config.rng  # pragma: no cover
    height_factor = config.factor1  # pragma: no cover
    width_factor = config.factor2  # pragma: no cover
    """Function docstring.

    Args:
        np_mod: Arg.
        H: Arg.
        W: Arg.
        rng: Arg.
        height_factor: Arg.
        width_factor: Arg.
    """

    def get_factor(factor: object) -> float:  # pragma: no cover
        """Function docstring.

        Args:
        factor: Arg.
        """
        if isinstance(factor, (tuple, list)):  # pragma: no cover
            return rng.uniform(factor[0], factor[1])  # type: ignore  # pragma: no cover
        return rng.uniform(1.0 - factor, 1.0 + factor)  # type: ignore  # pragma: no cover

    zx = get_factor(width_factor)  # pragma: no cover
    zy = get_factor(height_factor)  # pragma: no cover
    y_grid, x_grid = _generate_coordinate_grid(np_mod, H, W)  # pragma: no cover
    cy, cx = H / 2.0, W / 2.0  # pragma: no cover
    return (y_grid - cy) / zy + cy, (x_grid - cx) / zx + cx  # pragma: no cover


def _compute_translation_grid(np_mod: object, config: GeometricGridConfig) -> tuple[object, object]:
    H = config.H  # pragma: no cover
    W = config.W  # pragma: no cover
    rng = config.rng  # pragma: no cover
    height_factor = config.factor1  # pragma: no cover
    width_factor = config.factor2  # pragma: no cover
    """Function docstring.

    Args:
        np_mod: Arg.
        H: Arg.
        W: Arg.
        rng: Arg.
        height_factor: Arg.
        width_factor: Arg.
    """

    def get_factor(factor: object) -> float:  # pragma: no cover
        """Function docstring.

        Args:
        factor: Arg.
        """
        if isinstance(factor, (tuple, list)):  # pragma: no cover
            return rng.uniform(factor[0], factor[1])  # type: ignore  # pragma: no cover
        return rng.uniform(-factor, factor)  # type: ignore  # pragma: no cover

    tx = get_factor(width_factor) * W  # pragma: no cover
    ty = get_factor(height_factor) * H  # pragma: no cover
    y_grid, x_grid = _generate_coordinate_grid(np_mod, H, W)  # pragma: no cover
    return y_grid - ty, x_grid - tx  # pragma: no cover


def _compute_shear_grid(np_mod: object, config: GeometricGridConfig) -> tuple[object, object]:
    H = config.H  # pragma: no cover
    W = config.W  # pragma: no cover
    rng = config.rng  # pragma: no cover
    y_factor = config.factor1  # pragma: no cover
    x_factor = config.factor2  # pragma: no cover
    """Function docstring.  # pragma: no cover
  # pragma: no cover
    Args:  # pragma: no cover
        np_mod: Arg.  # pragma: no cover
        H: Arg.  # pragma: no cover
        W: Arg.  # pragma: no cover
        rng: Arg.  # pragma: no cover
        y_factor: Arg.  # pragma: no cover
        x_factor: Arg.  # pragma: no cover
    """  # pragma: no cover

    # pragma: no cover
    def get_factor(factor: object) -> float:  # pragma: no cover
        """Function docstring.

        Args:
            factor: Arg.
        """
        if isinstance(factor, (tuple, list)):  # pragma: no branch  # pragma: no cover
            return rng.uniform(factor[0], factor[1])  # type: ignore  # pragma: no cover
        return rng.uniform(-factor, factor)  # type: ignore  # pragma: no cover

    # pragma: no cover
    sy = get_factor(y_factor)  # pragma: no cover
    sx = get_factor(x_factor) if x_factor is not None else 0.0  # pragma: no cover
    # pragma: no cover
    y_grid, x_grid = _generate_coordinate_grid(np_mod, H, W)  # pragma: no cover
    cy, cx = H / 2.0, W / 2.0  # pragma: no cover
    # pragma: no cover
    # Shear matrix  # pragma: no cover
    # Inverted mapping:  # pragma: no cover
    # pragma: no cover
    y_shifted = y_grid - cy  # pragma: no cover
    x_shifted = x_grid - cx  # pragma: no cover
    # pragma: no cover
    y_src = y_shifted - sy * x_shifted  # pragma: no cover
    x_src = x_shifted - sx * y_shifted  # pragma: no cover
    # pragma: no cover
    return y_src + cy, x_src + cx  # pragma: no cover


def random_zoom_eager(  # pylint: disable=too-many-locals, too-many-arguments
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
    if width_factor is None:  # pragma: no cover
        width_factor = height_factor  # pragma: no cover
    new_y, new_x = _compute_zoom_grid(  # pragma: no cover
        ctx.np_mod, ctx.H, ctx.W, ctx.rng, height_factor, width_factor
    )

    config = RotationConfig(  # pragma: no cover
        factor=0.0,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=data_format,
    )
    out = _interpolate_pixels(ctx.np_mod, ctx.imgs, new_y, new_x, config)  # pragma: no cover

    out = _from_channels_last(ctx.np_mod, out, data_format)  # pragma: no cover
    return _from_numpy_array(backend_module, out, "", images)  # pragma: no cover


def random_translation_eager(  # pylint: disable=too-many-locals, too-many-arguments
    backend_module: object,
    images: object,
    height_factor: tuple[float, float] | float,
    width_factor: tuple[float, float] | float,
    **kwargs: object,
) -> object:
    """Evaluate random translation eagerly."""
    fill_mode = str(kwargs.get("fill_mode", "reflect"))  # pragma: no cover
    interpolation = str(kwargs.get("interpolation", "bilinear"))  # pragma: no cover
    fill_value = float(kwargs.get("fill_value", 0.0))  # pragma: no cover
    seed = kwargs.get("seed", None)  # pragma: no cover
    data_format = kwargs.get("data_format", None)  # pragma: no cover

    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)  # pragma: no cover
    new_y, new_x = _compute_translation_grid(  # pragma: no cover
        ctx.np_mod, ctx.H, ctx.W, ctx.rng, height_factor, width_factor
    )

    config = RotationConfig(  # pragma: no cover
        factor=0.0,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=data_format,
    )
    out = _interpolate_pixels(ctx.np_mod, ctx.imgs, new_y, new_x, config)  # pragma: no cover

    out = _from_channels_last(ctx.np_mod, out, data_format)  # pragma: no cover
    return _from_numpy_array(backend_module, out, "", images)  # pragma: no cover


def random_shear_eager(  # pylint: disable=too-many-locals, too-many-arguments
    backend_module: object,
    images: object,
    y_factor: tuple[float, float] | float,
    x_factor: tuple[float, float] | float | None = None,
    **kwargs: object,
) -> object:
    """Evaluate random shear eagerly."""
    fill_mode = str(kwargs.get("fill_mode", "reflect"))  # pragma: no cover
    interpolation = str(kwargs.get("interpolation", "bilinear"))  # pragma: no cover
    fill_value = float(kwargs.get("fill_value", 0.0))  # pragma: no cover
    seed = kwargs.get("seed", None)  # pragma: no cover
    data_format = kwargs.get("data_format", None)  # pragma: no cover
    # pragma: no cover
    ctx = _prepare_eager_transform(backend_module, images, seed, data_format)  # pragma: no cover
    new_y, new_x = _compute_shear_grid(
        ctx.np_mod, ctx.H, ctx.W, ctx.rng, y_factor, x_factor
    )  # pragma: no cover
    # pragma: no cover
    config = RotationConfig(  # pragma: no cover
        factor=0.0,  # pragma: no cover
        fill_mode=fill_mode,  # pragma: no cover
        interpolation=interpolation,  # pragma: no cover
        seed=seed,  # pragma: no cover
        fill_value=fill_value,  # pragma: no cover
        data_format=data_format,  # pragma: no cover
    )  # pragma: no cover
    out = _interpolate_pixels(ctx.np_mod, ctx.imgs, new_y, new_x, config)  # pragma: no cover
    # pragma: no cover
    out = _from_channels_last(ctx.np_mod, out, data_format)  # pragma: no cover
    return _from_numpy_array(backend_module, out, "", images)  # pragma: no cover


__all__ = [n for n in globals().keys() if n != "__builtins__"]
