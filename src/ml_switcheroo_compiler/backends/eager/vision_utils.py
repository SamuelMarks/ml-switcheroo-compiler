"""Vision geometry and coordinate mapping utilities for eager transformations."""

from __future__ import annotations

import types
import typing

import pydantic

from ml_switcheroo_compiler.backends.eager.utils import _to_channels_last, _to_numpy_array
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendArrayProtocol,
    EagerTensorProtocol,
    coerce_array,
    coerce_float,
    coerce_operand_list,
    unpack_pair,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.shape import OpValue as OpValue
from ml_switcheroo_compiler.ops.configs import PerspectiveConfig


@typing.runtime_checkable
class _NamedBackendModuleProtocol(typing.Protocol):
    """Protocol for backend modules exposing `__name__`."""

    __name__: str


def _resolve_backend_name(backend_module: BackendModuleProtocol | BackendArg | None) -> str:
    """Resolve the module name string of `backend_module`.

    Args:
        backend_module: Candidate backend module or None.

    Returns:
        str: Module name string, defaulting to `'numpy'`.
    """
    if isinstance(backend_module, _NamedBackendModuleProtocol):
        return str(backend_module.__name__)
    return "numpy"


def _resolve_backend_mod(
    backend_module: types.ModuleType | BackendModuleProtocol | None,
) -> BackendModuleProtocol:
    """Resolve the active backend module when `backend_module` is unspecified.

    Args:
        backend_module: Candidate backend module or None.

    Returns:
        BackendModuleProtocol: Resolved array module.
    """
    from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod

    return _resolve_eager_mod(backend_module)


class RandomCropConfig(pydantic.BaseModel):
    """Configuration class for random cropping parameters.

    Attributes:
        crop_h: Target crop height.
        crop_w: Target crop width.
        b: Batch index.
        c: Channel index.
        H: Original image height.
        W: Original image width.
        rng: Optional random number generator.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    crop_h: int
    crop_w: int
    b: int
    c: int
    H: int
    W: int
    rng: BackendArg | None = None


class GeometricGridConfig(pydantic.BaseModel):
    """Configuration parameters for geometric grid generation.

    Attributes:
        H: Grid height dimension.
        W: Grid width dimension.
        rng: Optional random number generator.
        factor1: Primary geometric perturbation factor.
        factor2: Secondary geometric perturbation factor.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    H: int
    W: int
    rng: BackendArg | None = None
    factor1: BackendArg | None = None
    factor2: BackendArg | None = None


class EagerTransformContext(pydantic.BaseModel):
    """Context holding parsed tensor dimensions and image arrays for eager execution.

    Attributes:
        B: Batch dimension size.
        H: Image height.
        W: Image width.
        C: Channel dimension size.
        name: Name identifier of the backend framework.
        np_mod: NumPy-compatible math module.
        imgs: Input image array converted to channels-last layout (B, H, W, C).
        rng: Random number generator instance.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    B: int
    H: int
    W: int
    C: int
    name: str
    np_mod: types.ModuleType | BackendModuleProtocol
    imgs: BackendArg = None
    rng: BackendArg = None


def _prepare_eager_transform(
    backend_module: types.ModuleType | BackendModuleProtocol | None,
    images: BackendArg,
    seed: BackendArg | None,
    data_format: BackendArg | None,
) -> EagerTransformContext:
    """Prepare and standardize input images and execution context for eager transformations.

    Args:
        backend_module: Execution backend module or framework reference.
        images: Input tensor or array representing batched or single images.
        seed: Random seed for deterministic operations.
        data_format: Layout specification ('channels_first' or 'channels_last').

    Returns:
        EagerTransformContext: Standardized context containing channels-last images.
    """
    np_mod = _resolve_backend_mod(backend_module)
    name: str = _resolve_backend_name(backend_module) if backend_module is not None else "numpy"

    if images is None:
        return EagerTransformContext(
            B=0,
            H=0,
            W=0,
            C=0,
            name=name,
            np_mod=np_mod,
            imgs=np_mod.empty((0, 0, 0, 0), dtype=np_mod.float32),
            rng=np_mod.random.default_rng(seed) if seed is not None else np_mod.random.default_rng(0),
        )

    arr = np_mod.asarray(_to_numpy_array(np_mod, images, name))

    # Standardize dimensions to 4D (B, H, W, C)
    if arr.ndim == 2:
        arr = arr[None, :, :, None]
    elif arr.ndim == 3:
        if data_format == "channels_first" or (arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4)):
            arr = np_mod.transpose(arr, (1, 2, 0))[None, ...]
        else:
            arr = arr[None, ...]
    elif arr.ndim == 4 and data_format == "channels_first":
        arr = np_mod.transpose(arr, (0, 2, 3, 1))

    b, h, w, c = arr.shape
    rng = np_mod.random.default_rng(seed) if seed is not None else np_mod.random.default_rng()
    return EagerTransformContext(B=b, H=h, W=w, C=c, name=name, np_mod=np_mod, imgs=arr, rng=rng)


class TransformInterpolationConfig(pydantic.BaseModel):
    """Configuration options for spatial coordinate interpolation.

    Attributes:
        order: Interpolation spline order (0=nearest, 1=bilinear, 3=bicubic).
        fill_value: Scalar constant for padding pixels outside valid bounds.
        new_y: Target Y spatial sampling coordinates.
        new_x: Target X spatial sampling coordinates.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    order: int
    fill_value: float
    new_y: BackendArg | None = None
    new_x: BackendArg | None = None


class ResizeContext(pydantic.BaseModel):
    """Configuration parameters for spatial image resizing grids.

    Attributes:
        H: Source image height.
        W: Source image width.
        new_H: Desired output height.
        new_W: Desired output width.
        align_corners: Whether grid extrema align with corner pixel centers.
        half_pixel_centers: Whether to offset pixel sampling by half a pixel.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    H: int
    W: int
    new_H: int
    new_W: int
    align_corners: bool
    half_pixel_centers: bool = False


class MapCoordsContext(pydantic.BaseModel):
    """Context holding arrays for 2D coordinate subpixel interpolation.

    Attributes:
        np_mod: NumPy-compatible math module.
        image: 2-D single channel image array of shape (H, W).
        y: Sampling Y-coordinates grid.
        x: Sampling X-coordinates grid.
        valid: Optional boolean mask specifying in-bounds coordinates.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    np_mod: types.ModuleType | BackendModuleProtocol | None = None
    image: BackendArg = None
    y: BackendArg = None
    x: BackendArg = None
    valid: BackendArg = None


def _map_coords_nearest(ctx: MapCoordsContext | None) -> BackendArg:
    """Perform nearest-neighbor spatial interpolation.

    Args:
        ctx: MapCoordsContext containing source image and coordinate grids.

    Returns:
        BackendArg: Interpolated 2D array sampled at integer-rounded coordinate locations.
    """
    if ctx is None or ctx.image is None or ctx.y is None or ctx.x is None:
        return 0

    np_mod = _resolve_backend_mod(ctx.np_mod)
    img = coerce_array(np_mod, ctx.image)
    h, w = img.shape[:2]

    eps = 1e-5
    y = coerce_array(np_mod, ctx.y)
    x = coerce_array(np_mod, ctx.x)

    in_bounds = (y >= -eps) & (y <= float(h - 1) + eps) & (x >= -eps) & (x <= float(w - 1) + eps)
    y_clamped = np_mod.clip(y, 0.0, float(h - 1))
    x_clamped = np_mod.clip(x, 0.0, float(w - 1))

    y_round = np_mod.rint(y_clamped).astype(int)
    x_round = np_mod.rint(x_clamped).astype(int)

    sampled = img[y_round, x_round]
    if ctx.valid is not None:
        in_bounds = in_bounds & coerce_array(np_mod, ctx.valid)
    return np_mod.where(in_bounds, sampled, 0.0)


def _map_coords_bilinear(ctx: MapCoordsContext | None) -> BackendArg:
    """Perform bilinear spatial subpixel interpolation.

    Args:
        ctx: MapCoordsContext containing source image and subpixel coordinates.

    Returns:
        BackendArg: Bilinearly interpolated 2D array.
    """
    if ctx is None or ctx.image is None or ctx.y is None or ctx.x is None:
        return 0

    np_mod = _resolve_backend_mod(ctx.np_mod)
    img = coerce_array(np_mod, ctx.image)
    h, w = img.shape[:2]

    eps = 1e-5
    y = coerce_array(np_mod, ctx.y)
    x = coerce_array(np_mod, ctx.x)

    in_bounds = (y >= -eps) & (y <= float(h - 1) + eps) & (x >= -eps) & (x <= float(w - 1) + eps)

    y_clamped = np_mod.clip(y, 0.0, float(h - 1))
    x_clamped = np_mod.clip(x, 0.0, float(w - 1))

    y0 = np_mod.clip(np_mod.floor(y_clamped).astype(int), 0, max(int(h) - 2, 0))
    x0 = np_mod.clip(np_mod.floor(x_clamped).astype(int), 0, max(int(w) - 2, 0))
    y1 = np_mod.minimum(y0 + 1, h - 1)
    x1 = np_mod.minimum(x0 + 1, w - 1)

    ay = np_mod.clip(y_clamped - y0, 0.0, 1.0)
    ax = np_mod.clip(x_clamped - x0, 0.0, 1.0)

    w00 = (1.0 - ay) * (1.0 - ax)
    w01 = (1.0 - ay) * ax
    w10 = ay * (1.0 - ax)
    w11 = ay * ax

    ia = img[y0, x0]
    ib = img[y0, x1]
    ic = img[y1, x0]
    id_val = img[y1, x1]

    interpolated = w00 * ia + w01 * ib + w10 * ic + w11 * id_val

    if ctx.valid is not None:
        in_bounds = in_bounds & coerce_array(np_mod, ctx.valid)
    return np_mod.where(in_bounds, interpolated, 0.0)


def _cubic_kernel(t: BackendArg, a: float = -0.5, np_mod: BackendModuleProtocol | None = None) -> EagerTensorProtocol:
    """Evaluate Catmull-Rom cubic interpolation weights.

    Args:
        t: Distance offsets from grid points.
        a: Cubic spline tension parameter (typically -0.5 or -0.75).
        np_mod: Optional execution backend module.

    Returns:
        EagerTensorProtocol: Evaluated cubic weights.
    """
    mod = _resolve_backend_mod(np_mod)
    abs_t: EagerTensorProtocol = mod.abs(t)
    abs_t2 = abs_t * abs_t
    abs_t3 = abs_t2 * abs_t

    w1 = (a + 2.0) * abs_t3 - (a + 3.0) * abs_t2 + 1.0
    w2 = a * abs_t3 - 5.0 * a * abs_t2 + 8.0 * a * abs_t - 4.0 * a

    return mod.where(abs_t <= 1.0, w1, mod.where(abs_t < 2.0, w2, 0.0))


def _map_coords_bicubic(ctx: MapCoordsContext | None) -> BackendArg:
    """Perform bicubic Catmull-Rom spatial subpixel interpolation.

    Args:
        ctx: MapCoordsContext containing source image and subpixel coordinates.

    Returns:
        BackendArg: Bicubic interpolated 2D array.
    """
    if ctx is None or ctx.image is None or ctx.y is None or ctx.x is None:
        return 0

    np_mod = _resolve_backend_mod(ctx.np_mod)
    img = coerce_array(np_mod, ctx.image)
    h, w = img.shape[:2]

    y = coerce_array(np_mod, ctx.y)
    x = coerce_array(np_mod, ctx.x)

    in_bounds = (y >= 0.0) & (y <= float(h - 1)) & (x >= 0.0) & (x <= float(w - 1))

    y_floor = np_mod.floor(y).astype(int)
    x_floor = np_mod.floor(x).astype(int)

    out = np_mod.zeros_like(y, dtype=np_mod.float32)

    for m in range(-1, 3):
        ym = y_floor + m
        ym_c = np_mod.clip(ym, 0, h - 1)
        wy = _cubic_kernel(y - ym.astype(np_mod.float32), np_mod=np_mod)

        for n in range(-1, 3):
            xn = x_floor + n
            xn_c = np_mod.clip(xn, 0, w - 1)
            wx = _cubic_kernel(x - xn.astype(np_mod.float32), np_mod=np_mod)

            out = out + wy * wx * img[ym_c, xn_c]

    if ctx.valid is not None:
        in_bounds = in_bounds & coerce_array(np_mod, ctx.valid)
    return np_mod.where(in_bounds, out, 0.0)


def _np_map_coordinates(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    image: BackendArg,
    coords: BackendArg,
    order: int = 1,
    fill_value: float = 0.0,
) -> BackendArg:
    """Sample an image array at arbitrary continuous coordinates with specified order.

    Args:
        np_mod: NumPy-compatible math module.
        image: Source 2D or 3D image array.
        coords: Tuple or array of (y_coords, x_coords).
        order: Interpolation order (0: nearest, 1: bilinear, 3: bicubic).
        fill_value: Extrapolation fill value for coordinates outside image bounds.

    Returns:
        BackendArg: Resampled image array matching the spatial shape of coords.
    """
    if image is None or coords is None:
        return 0

    mod = _resolve_backend_mod(np_mod)
    img_arr = coerce_array(mod, image)
    cy, cx = unpack_pair(coords)
    y: EagerTensorProtocol = mod.asarray(cy, dtype=mod.float32)
    x: EagerTensorProtocol = mod.asarray(cx, dtype=mod.float32)

    h, w = img_arr.shape[:2]
    eps = 1e-5
    valid_mask = (y >= -eps) & (y <= float(h - 1) + eps) & (x >= -eps) & (x <= float(w - 1) + eps)
    ctx = MapCoordsContext(np_mod=mod, image=img_arr, y=y, x=x, valid=valid_mask)

    if order == 0:
        res = _map_coords_nearest(ctx)
    elif order == 3:
        res = _map_coords_bicubic(ctx)
    else:
        res = _map_coords_bilinear(ctx)

    if fill_value != 0.0:
        res = mod.where(valid_mask, res, fill_value)
    return res


def _check_collinear_quad(points: BackendArg) -> bool:
    """Check if any triplet of four 2D points is collinear.

    Args:
        points: Array of shape (4, 2) containing point coordinates.

    Returns:
        bool: True if collinearity is detected among any 3 points, False otherwise.
    """
    pts = [unpack_pair(row) for row in coerce_operand_list(points)]
    triplets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for i, j, k in triplets:
        p1x, p1y = coerce_float(pts[i][0]), coerce_float(pts[i][1])
        p2x, p2y = coerce_float(pts[j][0]), coerce_float(pts[j][1])
        p3x, p3y = coerce_float(pts[k][0]), coerce_float(pts[k][1])
        area2 = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
        if abs(area2) < 1e-7:
            return True
    return False


def _compute_single_homography(
    np_mod: BackendModuleProtocol | types.ModuleType | None, src: BackendArg, dst: BackendArg
) -> BackendArg:
    """Compute 3x3 homography matrix from 4 corresponding point pairs via SVD.

    Args:
        np_mod: NumPy-compatible math module.
        src: 4x2 array of source coordinates (x, y).
        dst: 4x2 array of destination coordinates (u, v).

    Returns:
        BackendArg: 3x3 normalized homography matrix mapping src to dst.

    Raises:
        ValueError: If points are collinear or SVD produces a singular matrix.
    """
    if _check_collinear_quad(src) or _check_collinear_quad(dst):
        raise ValueError("Collinear points detected: 3 points lie on the same line in perspective transform")

    mod = np_mod
    src_arr = coerce_array(mod, src)
    dst_arr = coerce_array(mod, dst)
    # Direct Linear Transformation (DLT) matrix A (8 x 9)
    A = adapt_backend_module(mod).require_symbol("zeros")(
        (8, 9), dtype=adapt_backend_module(mod).require_symbol("float64")
    )
    for i in range(4):
        x = coerce_float(src_arr[i, 0])
        y = coerce_float(src_arr[i, 1])
        u = coerce_float(dst_arr[i, 0])
        v = coerce_float(dst_arr[i, 1])

        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    _, s, vh = adapt_backend_module(mod).require_symbol("linalg").svd(A)

    # Validate non-singularity: rank must be 8 (second smallest singular value > 0)
    if s[6] < 1e-8 or (s[0] / max(float(s[6]), 1e-12)) > 1e9:
        raise ValueError("Singular matrix encountered in perspective transform estimation")

    h = vh[-1]
    if abs(h[-1]) > 1e-12:
        h = h / h[-1]
    return h.reshape(3, 3).astype(adapt_backend_module(mod).require_symbol("float32"))


def _compute_perspective_matrix(
    np_mod: BackendModuleProtocol | types.ModuleType | None, src: BackendArg, dst: BackendArg
) -> BackendArg:
    """Compute 3x3 homography matrices for single or batched point correspondences.

    Args:
        np_mod: NumPy-compatible math module.
        src: Source points array of shape (4, 2) or (B, 4, 2).
        dst: Destination points array of shape (4, 2) or (B, 4, 2).

    Returns:
        BackendArg: 3x3 or (B, 3, 3) projective transformation matrices.

    Raises:
        ValueError: If input points are collinear or singular.
    """
    if src is None or dst is None:
        return 0

    mod = _resolve_backend_mod(np_mod)
    src_arr = mod.asarray(src, dtype=mod.float64)
    dst_arr = mod.asarray(dst, dtype=mod.float64)

    if src_arr.ndim == 2:
        return _compute_single_homography(mod, src_arr, dst_arr)

    b_count = src_arr.shape[0]
    matrices = [_compute_single_homography(mod, src_arr[b], dst_arr[b]) for b in range(b_count)]
    return mod.stack(matrices, axis=0)


def _generate_perspective_coords(
    np_mod: BackendModuleProtocol | types.ModuleType | None, h_batch: BackendArg, coords: BackendArg
) -> BackendArg:
    """Generate source x and y coordinates for a given batch from homography matrix.

    Args:
        np_mod: NumPy-compatible math module.
        h_batch: Homography matrix of shape (3, 3) or (B, 3, 3).
        coords: Meshgrid of homogeneous pixel coordinates of shape (H, W, 3).

    Returns:
        BackendArg: Tuple of (ys, xs) backward-mapped coordinate grids.
    """
    if h_batch is None or coords is None:
        return 0

    mod = _resolve_backend_mod(np_mod)
    h = mod.asarray(h_batch, dtype=mod.float32)
    c = mod.asarray(coords, dtype=mod.float32)

    if h.ndim == 2:
        flat_c = c.reshape(-1, 3)
        transformed = flat_c @ h.T
        denom = mod.where(mod.abs(transformed[:, 2]) < 1e-12, 1e-12, transformed[:, 2])
        xs = (transformed[:, 0] / denom).reshape(c.shape[0], c.shape[1])
        ys = (transformed[:, 1] / denom).reshape(c.shape[0], c.shape[1])
        return (ys, xs)

    # Batched homographies (B, 3, 3)
    b_count = h.shape[0]
    ys_list = []
    xs_list = []
    for b in range(b_count):
        sub_coords = unpack_pair(_generate_perspective_coords(mod, h[b], c))
        ys_list.append(sub_coords[0])
        xs_list.append(sub_coords[1])
    return (mod.stack(ys_list, axis=0), mod.stack(xs_list, axis=0))


def _generate_perspective_grid(np_mod: BackendModuleProtocol | types.ModuleType | None, H: int, W: int) -> BackendArg:
    """Generate homogeneous 2D meshgrid spanning spatial dimensions (H, W).

    Args:
        np_mod: NumPy-compatible math module.
        H: Target grid height.
        W: Target grid width.

    Returns:
        BackendArg: Array of shape (H, W, 3) containing (x, y, 1.0) coordinates.
    """
    if H <= 0 or W <= 0:
        return 0

    mod = _resolve_backend_mod(np_mod)
    y, x = mod.meshgrid(
        mod.arange(H, dtype=mod.float32),
        mod.arange(W, dtype=mod.float32),
        indexing="ij",
    )
    ones = mod.ones_like(x)
    return mod.stack([x, y, ones], axis=-1)


class PerspectiveContext(pydantic.BaseModel):
    """Context holding batch index and coordinate grids for perspective transformation.

    Attributes:
        b: Current batch index.
        coords: Tuple of (ys, xs) sampling coordinates.
        h: Homography matrix for this batch item.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    b: int
    coords: BackendArg | None = None
    h: BackendArg | None = None


class PerspectiveChannelContext(pydantic.BaseModel):
    """Context bundle for applying perspective transformation to a single image channel.

    Attributes:
        ctx: PerspectiveContext containing batch indices and coordinates.
        config: PerspectiveConfig governing interpolation and fill values.
        np_mod: NumPy-compatible math module.
        imgs: Input batched images array of shape (B, H, W, C).
        out: Target output array of shape (B, H, W, C).
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    ctx: PerspectiveContext
    config: PerspectiveConfig
    np_mod: BackendModuleProtocol | None = None
    imgs: BackendArrayProtocol | None = None
    out: BackendArrayProtocol | None = None


def _apply_perspective_channel(pctx: PerspectiveChannelContext | None) -> BackendArg:
    """Apply perspective warping across channels for an individual batch element.

    Args:
        pctx: PerspectiveChannelContext bundling arrays, coordinates, and configurations.

    Returns:
        BackendArg: Updated output array with warped image channels.
    """
    if pctx is None or pctx.imgs is None or pctx.out is None or pctx.ctx is None:
        return 0

    np_mod = _resolve_backend_mod(pctx.np_mod)
    b = pctx.ctx.b
    coords = pctx.ctx.coords
    order = 1 if adapt_backend_module(pctx.config).get_symbol("interpolation", "bilinear") == "bilinear" else 0
    fill_value = coerce_float(adapt_backend_module(pctx.config).get_symbol("fill_value", 0.0))

    c_count = pctx.imgs.shape[3]
    for c in range(c_count):
        pctx.out[b, :, :, c] = _np_map_coordinates(
            np_mod,
            pctx.imgs[b, :, :, c],
            coords,
            order=order,
            fill_value=fill_value,
        )
    return pctx.out


def _apply_perspective_batch(
    np_mod: BackendModuleProtocol | types.ModuleType | None,
    imgs: BackendArg,
    h: BackendArg,
    config: PerspectiveConfig | None,
) -> BackendArg:
    """Apply perspective transform across a batched image array using homography matrices.

    Args:
        np_mod: NumPy-compatible math module.
        imgs: Input batched image array of shape (B, H, W, C).
        h: Backward-mapping homography matrix (3, 3) or (B, 3, 3).
        config: PerspectiveConfig controlling interpolation method and padding values.

    Returns:
        BackendArg: Transformed batched image array of shape (B, H, W, C), or `0` on `None` inputs.
    """
    if imgs is None or h is None or config is None:
        return 0

    mod = _resolve_backend_mod(np_mod)

    imgs_arr = coerce_array(mod, imgs)
    h_arr = coerce_array(mod, h)
    b_count, h_dim, w_dim, _ = imgs_arr.shape
    grid = _generate_perspective_grid(mod, h_dim, w_dim)
    out = mod.zeros_like(imgs_arr)

    for b in range(b_count):
        h_item = h_arr[b] if h_arr.ndim == 3 else h_arr
        coords = _generate_perspective_coords(mod, h_item, grid)
        ctx = PerspectiveContext(b=b, coords=coords, h=h_item)
        pctx = PerspectiveChannelContext(ctx=ctx, config=config, np_mod=mod, imgs=imgs_arr, out=out)
        _apply_perspective_channel(pctx)

    return out


__all__ = [
    "EagerTransformContext",
    "GeometricGridConfig",
    "MapCoordsContext",
    "PerspectiveChannelContext",
    "PerspectiveConfig",
    "PerspectiveContext",
    "RandomCropConfig",
    "ResizeContext",
    "TransformInterpolationConfig",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_apply_perspective_batch",
    "_apply_perspective_channel",
    "_compute_perspective_matrix",
    "_generate_perspective_coords",
    "_generate_perspective_grid",
    "_map_coords_bilinear",
    "_map_coords_nearest",
    "_np_map_coordinates",
    "_prepare_eager_transform",
    "_to_channels_last",
    "_to_numpy_array",
    "annotations",
]
