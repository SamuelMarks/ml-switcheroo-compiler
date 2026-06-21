# ruff: noqa: E402, D100, D101
"""Vision utilities."""

from __future__ import annotations

from dataclasses import dataclass

from ml_switcheroo_compiler.ops.configs import ElasticConfig, PerspectiveConfig, ResizeOptions


@dataclass
class EagerTransformContext:  # pylint: disable=too-many-instance-attributes
    np_mod: object
    rng: object
    imgs: object
    B: int
    H: int
    W: int
    C: int
    name: str


def _prepare_eager_transform(
    backend_module: object, images: object, seed: object, data_format: object
) -> EagerTransformContext:
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(seed)
    imgs = _to_channels_last(np_mod, imgs, data_format)
    B, H, W, C = imgs.shape  # type: ignore
    return EagerTransformContext(np_mod, rng, imgs, B, H, W, C, name)


@dataclass
class TransformInterpolationConfig:
    new_y: object
    new_x: object
    order: int
    fill_value: float


@dataclass
class ResizeContext:
    H: int
    W: int
    new_H: int
    new_W: int
    align_corners: bool


from ml_switcheroo_compiler.backends.eager.utils import (
    _from_channels_last,
    _from_numpy_array,
    _to_channels_last,
    _to_numpy_array,
)


def _map_coords_nearest(
    np_mod: object, image: object, y: object, x: object, valid: object
) -> object:
    y_idx = np_mod.round(y[valid]).astype(np_mod.int32)
    x_idx = np_mod.round(x[valid]).astype(np_mod.int32)
    y_idx = np_mod.clip(y_idx, 0, image.shape[0] - 1)
    x_idx = np_mod.clip(x_idx, 0, image.shape[1] - 1)
    return image[y_idx, x_idx]


def _map_coords_bilinear(
    np_mod: object, image: object, y: object, x: object, valid: object
) -> object:
    y0 = np_mod.floor(y[valid]).astype(np_mod.int32)
    x0 = np_mod.floor(x[valid]).astype(np_mod.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    y0 = np_mod.clip(y0, 0, image.shape[0] - 1)
    x0 = np_mod.clip(x0, 0, image.shape[1] - 1)
    y1 = np_mod.clip(y1, 0, image.shape[0] - 1)
    x1 = np_mod.clip(x1, 0, image.shape[1] - 1)
    dy = y[valid] - y0
    dx = x[valid] - x0
    w00 = (1 - dy) * (1 - dx)
    w01 = (1 - dy) * dx
    w10 = dy * (1 - dx)
    w11 = dy * dx
    return image[y0, x0] * w00 + image[y0, x1] * w01 + image[y1, x0] * w10 + image[y1, x1] * w11


def _np_map_coordinates(
    np_mod: object, image: object, coords: object, order: int = 1, fill_value: float = 0.0
) -> object:
    y, x = coords[0], coords[1]
    out = np_mod.full(y.shape, fill_value, dtype=image.dtype)
    valid = (y >= 0) & (y <= image.shape[0] - 1) & (x >= 0) & (x <= image.shape[1] - 1)
    if order == 0:
        out[valid] = _map_coords_nearest(np_mod, image, y, x, valid)
    else:
        out[valid] = _map_coords_bilinear(np_mod, image, y, x, valid)
    return out


def _compute_perspective_matrix(np_mod: object, src: object, dst: object) -> object:
    A = np_mod.zeros((*dst.shape[:-2], 8, 8), dtype=np_mod.float32)
    B = np_mod.zeros((*dst.shape[:-2], 8), dtype=np_mod.float32)

    for i in range(4):
        u = dst[..., i, 0]
        v = dst[..., i, 1]
        x = src[..., i, 0]
        y = src[..., i, 1]

        A[..., i * 2, 0] = u
        A[..., i * 2, 1] = v
        A[..., i * 2, 2] = 1
        A[..., i * 2, 6] = -x * u
        A[..., i * 2, 7] = -x * v

        A[..., i * 2 + 1, 3] = u
        A[..., i * 2 + 1, 4] = v
        A[..., i * 2 + 1, 5] = 1
        A[..., i * 2 + 1, 6] = -y * u
        A[..., i * 2 + 1, 7] = -y * v

        B[..., i * 2] = x
        B[..., i * 2 + 1] = y

    h = np_mod.linalg.solve(A, B)
    return h


def _generate_perspective_coords(
    np_mod: object, h_batch: object, coords: object
) -> tuple[object, object]:
    """Generate source x and y coordinates for a given batch from homography matrix."""
    H_mat = np_mod.concatenate([h_batch, [1.0]]).reshape(3, 3)
    t_coords = coords @ H_mat.T
    t_coords = t_coords / t_coords[..., 2:3]
    return t_coords[..., 0], t_coords[..., 1]


def _generate_perspective_grid(np_mod: object, H: int, W: int) -> object:
    y_grid, x_grid = np_mod.meshgrid(np_mod.arange(H), np_mod.arange(W), indexing="ij")
    y_grid = y_grid.astype(np_mod.float32)
    x_grid = x_grid.astype(np_mod.float32)
    ones = np_mod.ones_like(x_grid)
    return np_mod.stack([x_grid, y_grid, ones], axis=-1)


@dataclass
class PerspectiveContext:
    coords: object
    h: object
    b: int


def _apply_perspective_channel(  # pylint: disable=too-many-arguments
    np_mod: object, imgs: object, out: object, ctx: PerspectiveContext, config: PerspectiveConfig
) -> None:
    src_x, src_y = _generate_perspective_coords(np_mod, ctx.h, ctx.coords)
    C = imgs.shape[-1]
    order = 1 if config.interpolation == "bilinear" else 0
    for c in range(C):
        channel = imgs[ctx.b, ..., c]
        res = _np_map_coordinates(
            np_mod, channel, [src_y, src_x], order=order, fill_value=config.fill_value
        )
        out[ctx.b, ..., c] = res


def _apply_perspective_batch(
    np_mod: object, imgs: object, h: object, config: PerspectiveConfig
) -> object:
    """Apply perspective transform to a batched image array."""
    B_sz, H, W, C = imgs.shape
    out = np_mod.zeros_like(imgs)
    coords = _generate_perspective_grid(np_mod, H, W)

    for b in range(B_sz):
        _apply_perspective_channel(np_mod, imgs, out, coords, h=h[b], b=b, config=config)
    return out


def perspective_transform_eager(
    backend_module: object,
    images: object,
    start_points: object,
    end_points: object,
    config: PerspectiveConfig,
) -> object:
    """Evaluate perspective transform eagerly."""
    name = getattr(backend_module, "__name__", "")

    config.interpolation = getattr(config, "interpolation", "bilinear")
    config.fill_value = getattr(config, "fill_value", 0.0)
    config.data_format = getattr(config, "data_format", None)

    if name == "keras.ops":
        return backend_module.image.perspective_transform(
            images,
            start_points,
            end_points,
            config.interpolation,
            config.fill_value,
            config.data_format,
        )

    np_mod = __import__("numpy")

    imgs = _to_numpy_array(np_mod, images, name)
    src = _to_numpy_array(np_mod, start_points, name)
    dst = _to_numpy_array(np_mod, end_points, name)

    imgs = _to_channels_last(np_mod, imgs, config.data_format)

    h = _compute_perspective_matrix(np_mod, src, dst)

    has_batch = imgs.ndim == 4
    if not has_batch:
        imgs = imgs[None, ...]
        h = h[None, ...]

    out = _apply_perspective_batch(np_mod, imgs, h, config)

    if not has_batch:
        out = out[0]

    out = _from_channels_last(np_mod, out, config.data_format)

    return _from_numpy_array(backend_module, out, name, images)


def _apply_elastic_batch(
    np_mod: object, imgs: object, config: TransformInterpolationConfig
) -> object:
    """Apply elastic coordinates across a batch."""
    B, H, W, C = imgs.shape
    out = np_mod.zeros_like(imgs)
    for b in range(B):
        for c in range(C):
            coords = [config.new_y[b], config.new_x[b]]
            out[b, ..., c] = _np_map_coordinates(
                np_mod, imgs[b, ..., c], coords, order=config.order, fill_value=config.fill_value
            )
    return out


def _compute_elastic_grid(
    np_mod: object, H: int, W: int, B: int, disp: object
) -> tuple[object, object]:
    y, x = np_mod.meshgrid(np_mod.arange(H), np_mod.arange(W), indexing="ij")
    y = np_mod.broadcast_to(y, (B, H, W))
    x = np_mod.broadcast_to(x, (B, H, W))
    dy = disp[..., 0]
    dx = disp[..., 1]
    return y + dy, x + dx


def elastic_transform_eager(  # pylint: disable=too-many-locals
    backend_module: object,
    images: object,
    displacement: object,
    config: ElasticConfig,
) -> object:
    """Evaluate elastic transform eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    imgs = _to_numpy_array(np_mod, images, name)
    disp = _to_numpy_array(np_mod, displacement, name)

    original_ndim = imgs.ndim
    if original_ndim == 3:
        imgs = imgs[None, ...]
        disp = disp[None, ...]

    imgs = _to_channels_last(np_mod, imgs, config.data_format)

    B, H, W, C = imgs.shape

    new_y, new_x = _compute_elastic_grid(np_mod, H, W, B, disp)

    order = 1 if config.interpolation == "bilinear" else 0

    config = TransformInterpolationConfig(
        new_y=new_y, new_x=new_x, order=order, fill_value=config.fill_value
    )
    out = _apply_elastic_batch(np_mod, imgs, config)

    out = _from_channels_last(np_mod, out, config.data_format)

    if original_ndim == 3:
        out = out[0]

    return _from_numpy_array(backend_module, out, name, images)


def _get_resize_interpolation_order(interpolation: str) -> int:
    """Get scipy ndimage order for interpolation string."""
    if interpolation == "nearest":
        return 0
    elif interpolation in ("bicubic", "lanczos3"):
        return 3
    return 1


def _compute_resize_grid(np_mod: object, ctx: ResizeContext) -> tuple[object, object]:
    """Compute the sampling grid for resize operation."""
    H, W, new_H, new_W, align_corners = ctx.H, ctx.W, ctx.new_H, ctx.new_W, ctx.align_corners
    if align_corners:
        scale_y = (H - 1) / max(1, new_H - 1) if new_H > 1 else 0
        scale_x = (W - 1) / max(1, new_W - 1) if new_W > 1 else 0
    else:
        scale_y = H / new_H
        scale_x = W / new_W

    y_coords = np_mod.arange(new_H) * scale_y
    x_coords = np_mod.arange(new_W) * scale_x
    if not align_corners:
        y_coords += 0.5 * scale_y - 0.5
        x_coords += 0.5 * scale_x - 0.5

    return np_mod.meshgrid(y_coords, x_coords, indexing="ij")


def _apply_resize_batch(
    np_mod: object, imgs: object, out: object, coords: tuple[object, object], order: int
) -> None:
    B = imgs.shape[0]
    C = imgs.shape[-1]
    yy, xx = coords
    for b in range(B):
        for c in range(C):
            out[b, ..., c] = _np_map_coordinates(np_mod, imgs[b, ..., c], [yy, xx], order=order)


def resize_eager(  # pylint: disable=too-many-locals
    backend_module: object,
    images: object,
    size: tuple[int, int],
    config: ResizeOptions,
) -> object:
    """Evaluate resize eagerly."""
    name = getattr(backend_module, "__name__", "")
    np_mod = __import__("numpy")

    if name == "keras.ops":
        import tensorflow as tf

        images_tf = tf.convert_to_tensor(images)
        res = tf.image.resize(images_tf, size, method=config.interpolation, antialias=True)
        return backend_module.convert_to_tensor(res)

    imgs = _to_numpy_array(np_mod, images, name)

    B, H, W, C = imgs.shape
    new_H, new_W = size

    order = _get_resize_interpolation_order(config.interpolation)
    out = np_mod.zeros((B, new_H, new_W, C), dtype=np_mod.float32)

    ctx = ResizeContext(H=H, W=W, new_H=new_H, new_W=new_W, align_corners=config.align_corners)
    yy, xx = _compute_resize_grid(np_mod, ctx)

    _apply_resize_batch(np_mod, imgs, out, (yy, xx), order)

    return _from_numpy_array(backend_module, out, name, images)


def random_flip_eager(
    backend_module: object, images: object, mode: str, seed: object = None
) -> object:
    """Evaluate random flip eagerly."""
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(seed)

    if mode in ("horizontal", "horizontal_and_vertical"):
        if rng.random() > 0.5:
            imgs = np_mod.flip(imgs, axis=-2)  # width is -2 if shape is (B, H, W, C)
    if mode in ("vertical", "horizontal_and_vertical"):
        if rng.random() > 0.5:
            imgs = np_mod.flip(imgs, axis=-3)  # height is -3

    return _from_numpy_array(backend_module, imgs, name, images)


from dataclasses import dataclass


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
    angle_rad = rng.uniform(-factor, factor) * np_mod.pi / 180.0
    cos_a = np_mod.cos(angle_rad)
    sin_a = np_mod.sin(angle_rad)
    y_grid, x_grid = _generate_coordinate_grid(np_mod, H, W)
    cy, cx = H / 2.0, W / 2.0
    return _apply_affine_transform(y_grid, x_grid, AffineTransformParams(cos_a, sin_a, cx, cy))


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
    B: int,
    H: int,
    W: int,
    C: int,
    height: int,
    width: int,
    rng: object,
) -> object:
    out = np_mod.zeros((B, height, width, C), dtype=imgs.dtype)  # type: ignore
    for b in range(B):
        y_start = rng.integers(0, H - height + 1) if H >= height else 0  # type: ignore
        x_start = rng.integers(0, W - width + 1) if W >= width else 0  # type: ignore
        y_end = min(y_start + height, H)
        x_end = min(x_start + width, W)
        cropped = imgs[b, y_start:y_end, x_start:x_end, :]  # type: ignore
        pad_y = height - cropped.shape[0]
        pad_x = width - cropped.shape[1]
        if pad_y > 0 or pad_x > 0:
            cropped = np_mod.pad(cropped, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant")  # type: ignore
        out[b] = cropped  # type: ignore
    return out


def random_crop_eager(  # pylint: disable=too-many-locals
    backend_module: object, images: object, size: tuple, seed: object = None
) -> object:
    """Evaluate random crop eagerly."""
    np_mod = __import__("numpy")
    name = getattr(backend_module, "__name__", "")
    imgs = _to_numpy_array(np_mod, images, name)
    rng = np_mod.random.default_rng(seed)

    B, H, W, C = imgs.shape
    new_H, new_W = size

    if H <= new_H and W <= new_W:
        return images

    start_h = rng.integers(0, H - new_H + 1) if H > new_H else 0
    start_w = rng.integers(0, W - new_W + 1) if W > new_W else 0

    out = imgs[:, start_h : start_h + new_H, start_w : start_w + new_W, :]
    return _from_numpy_array(backend_module, out, name, images)


def _compute_zoom_grid(
    np_mod: object, H: int, W: int, rng: object, height_factor: object, width_factor: object
) -> tuple[object, object]:
    def get_factor(factor: object) -> float:
        if isinstance(factor, (tuple, list)):
            return rng.uniform(factor[0], factor[1])  # type: ignore
        return rng.uniform(1.0 - factor, 1.0 + factor)  # type: ignore

    zx = get_factor(width_factor)
    zy = get_factor(height_factor)
    y_grid, x_grid = _generate_coordinate_grid(np_mod, H, W)
    cy, cx = H / 2.0, W / 2.0
    return (y_grid - cy) / zy + cy, (x_grid - cx) / zx + cx


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
    if width_factor is None:
        width_factor = height_factor
    new_y, new_x = _compute_zoom_grid(
        ctx.np_mod, ctx.H, ctx.W, ctx.rng, height_factor, width_factor
    )

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


def _compute_translation_grid(
    np_mod: object, H: int, W: int, rng: object, height_factor: object, width_factor: object
) -> tuple[object, object]:
    def get_factor(factor: object) -> float:
        if isinstance(factor, (tuple, list)):
            return rng.uniform(factor[0], factor[1])  # type: ignore
        return rng.uniform(-factor, factor)  # type: ignore

    tx = get_factor(width_factor) * W
    ty = get_factor(height_factor) * H
    y_grid, x_grid = _generate_coordinate_grid(np_mod, H, W)
    return y_grid - ty, x_grid - tx


def random_translation_eager(  # pylint: disable=too-many-locals, too-many-arguments
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
    new_y, new_x = _compute_translation_grid(
        ctx.np_mod, ctx.H, ctx.W, ctx.rng, height_factor, width_factor
    )

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
