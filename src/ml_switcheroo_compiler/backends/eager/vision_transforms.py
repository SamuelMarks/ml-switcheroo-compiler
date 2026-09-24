"""Vision geometric transformations and spatial upsampling eager kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager.utils import (
    _from_numpy_array,
    _to_numpy_array,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.ops.configs import ElasticConfig, PerspectiveConfig, ResizeOptions

from .vision_utils import (
    ResizeContext,
    TransformInterpolationConfig,
    _apply_perspective_batch,
    _compute_perspective_matrix,
    _np_map_coordinates,
)


def _standardize_image_input(
    arr: np.ndarray,
    data_format: str | None,
) -> tuple[np.ndarray, bool]:
    """Standardize input array to 4D channels-last (B, H, W, C).

    Args:
        arr: Input array of 2D, 3D, or 4D shape.
        data_format: Optional data format string ('channels_first' or 'channels_last').

    Returns:
        tuple[np.ndarray, bool]: 4D array and flag indicating if original input was 3D.
    """
    if arr.ndim == 2:
        return arr[None, :, :, None], False
    if arr.ndim == 3:
        if data_format == "channels_first" or (arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4)):
            return np.transpose(arr, (1, 2, 0))[None, ...], True
        return arr[None, ...], True
    if arr.ndim == 4 and data_format == "channels_first":
        return np.transpose(arr, (0, 2, 3, 1)), False
    return arr, False


def _invert_homography(np_mod: Any, h_fwd: np.ndarray) -> np.ndarray:
    """Invert single or batched homography matrices.

    Args:
        np_mod: NumPy-compatible execution module.
        h_fwd: 3x3 or (B, 3, 3) forward homography matrix.

    Returns:
        np.ndarray: Inverse homography matrix mapping destination to source.
    """
    if h_fwd.ndim == 2:
        return np_mod.linalg.inv(h_fwd)
    return np_mod.stack([np_mod.linalg.inv(h_fwd[b]) for b in range(h_fwd.shape[0])], axis=0)


def perspective_transform_eager(
    backend_module: Any,
    images: Any,
    start_points: object,
    end_points: object,
    config: PerspectiveConfig,
) -> Any:
    """Evaluate perspective_transform_eager operation.

    Args:
        backend_module: The execution backend module.
        images: Input images tensor or array.
        start_points: Coordinates of 4 points in source images (4, 2) or (B, 4, 2).
        end_points: Coordinates of 4 points in destination images (4, 2) or (B, 4, 2).
        config (PerspectiveConfig): Configuration parameters including interpolation and fill.

    Returns:
        Any: Transformed images matching input backend representation.
    """
    if images is None:
        return 0

    np_mod = np if backend_module is None else getattr(backend_module, "numpy", np)
    name: str = getattr(backend_module, "__name__", "numpy") if backend_module is not None else "numpy"

    arr = np_mod.asarray(_to_numpy_array(np_mod, images, name))

    data_format = getattr(config, "data_format", None)
    arr, was_3d = _standardize_image_input(arr, data_format)

    if start_points is None or end_points is None:
        out = arr.copy()
    else:
        h_fwd = _compute_perspective_matrix(np_mod, start_points, end_points)
        h_inv = _invert_homography(np_mod, h_fwd)
        out = _apply_perspective_batch(np_mod, arr, h_inv, config)

    if data_format == "channels_first":
        out = np.transpose(out, (0, 3, 1, 2))
    if was_3d and out.shape[0] == 1:
        out = out[0]

    return _from_numpy_array(backend_module, out, name, images)


def _apply_elastic_batch(np_mod: Any, imgs: Any, config: TransformInterpolationConfig) -> Any:
    """Apply elastic coordinates across a batch.

    Args:
        np_mod: NumPy-compatible execution module.
        imgs: Input batched images array of shape (B, H, W, C).
        config (TransformInterpolationConfig): Configuration containing target coordinates and order.

    Returns:
        Any: Warped batched images array.
    """
    if imgs is None or config is None:
        return 0

    np_mod = np if np_mod is None else np_mod
    b_count, _, _, c_count = imgs.shape
    out = np_mod.zeros_like(imgs)

    y_coords = config.new_y
    x_coords = config.new_x

    for b in range(b_count):
        y_idx = min(b, y_coords.shape[0] - 1) if getattr(y_coords, "ndim", 2) == 3 else 0
        x_idx = min(b, x_coords.shape[0] - 1) if getattr(x_coords, "ndim", 2) == 3 else 0
        y_b = y_coords[y_idx] if getattr(y_coords, "ndim", 2) == 3 else y_coords
        x_b = x_coords[x_idx] if getattr(x_coords, "ndim", 2) == 3 else x_coords
        for c in range(c_count):
            out[b, :, :, c] = _np_map_coordinates(
                np_mod,
                imgs[b, :, :, c],
                (y_b, x_b),
                order=config.order,
                fill_value=config.fill_value,
            )
    return out


@dataclass
class ElasticGridContext:
    """Context holding spatial dimensions and displacement fields for elastic warping.

    Attributes:
        np_mod: NumPy-compatible execution module.
        H: Image height.
        W: Image width.
        B: Batch size.
        disp: Displacement field array or numeric constant.
    """

    np_mod: Any
    H: int
    W: int
    B: int
    disp: object


def _compute_elastic_grid(ctx: ElasticGridContext) -> Any:
    """Evaluate _compute_elastic_grid operation.

    Args:
        ctx (ElasticGridContext): Elastic distortion context.

    Returns:
        Any: Tuple of (new_y, new_x) distorted sampling coordinates.
    """
    if ctx is None:
        return 0

    np_mod = np if ctx.np_mod is None else ctx.np_mod
    h = ctx.H
    w = ctx.W

    y_grid, x_grid = np_mod.meshgrid(
        np_mod.arange(h, dtype=np_mod.float32),
        np_mod.arange(w, dtype=np_mod.float32),
        indexing="ij",
    )

    if ctx.disp is None or (isinstance(ctx.disp, (int, float)) and ctx.disp == 0):
        return (y_grid, x_grid)

    disp = np_mod.asarray(ctx.disp, dtype=np_mod.float32)
    if disp.ndim == 3:
        if disp.shape[0] == 2:
            dy, dx = disp[0], disp[1]
        else:
            dy, dx = disp[..., 0], disp[..., 1]
    elif disp.ndim == 4:
        if disp.shape[1] == 2:
            dy, dx = disp[:, 0], disp[:, 1]
        else:
            dy, dx = disp[..., 0], disp[..., 1]
    else:
        dy, dx = np_mod.zeros((h, w), dtype=np_mod.float32), np_mod.zeros((h, w), dtype=np_mod.float32)

    return (y_grid + dy, x_grid + dx)


def elastic_transform_eager(
    backend_module: Any,
    images: Any,
    displacement: object,
    config: ElasticConfig,
) -> Any:
    """Evaluate elastic_transform_eager operation.

    Args:
        backend_module: Execution backend module.
        images: Input images tensor or array.
        displacement: 2D/3D/4D spatial displacement field.
        config (ElasticConfig): Configuration containing interpolation and fill options.

    Returns:
        Any: Elastically distorted images.
    """
    if images is None:
        return 0

    np_mod = np if backend_module is None else getattr(backend_module, "numpy", np)
    name: str = getattr(backend_module, "__name__", "numpy") if backend_module is not None else "numpy"

    arr = np_mod.asarray(_to_numpy_array(np_mod, images, name))

    data_format = getattr(config, "data_format", None)
    arr, was_3d = _standardize_image_input(arr, data_format)

    b, h, w, _ = arr.shape
    if displacement is None:
        out = arr.copy()
    else:
        interp_name = getattr(config, "interpolation", "bilinear")
        order = _get_resize_interpolation_order(interp_name)
        fill_val = float(getattr(config, "fill_value", 0.0))

        ctx = ElasticGridContext(np_mod=np_mod, H=h, W=w, B=b, disp=displacement)
        new_y, new_x = _compute_elastic_grid(ctx)
        cfg = TransformInterpolationConfig(order=order, fill_value=fill_val, new_y=new_y, new_x=new_x)
        out = _apply_elastic_batch(np_mod, arr, cfg)

    if data_format == "channels_first":
        out = np.transpose(out, (0, 3, 1, 2))
    if was_3d and out.shape[0] == 1:
        out = out[0]

    return _from_numpy_array(backend_module, out, name, images)


def _get_resize_interpolation_order(interpolation: str) -> int:
    """Get scipy ndimage order for interpolation string.

    Args:
        interpolation (str): The interpolation parameter.

    Returns:
        int: Result.
    """
    if interpolation == "nearest":
        return 0
    elif interpolation in ("bicubic", "lanczos3"):
        return 3
    return 1


def _compute_resize_grid(np_mod: Any, ctx: ResizeContext) -> Any:
    """Evaluate _compute_resize_grid operation.

    Args:
        np_mod: NumPy-compatible execution module.
        ctx (ResizeContext): Spatial resizing context.

    Returns:
        Any: Tuple of (y_grid, x_grid) continuous coordinates.
    """
    if ctx is None:
        return 0

    np_mod = np if np_mod is None else np_mod
    h, w = ctx.H, ctx.W
    new_h, new_w = ctx.new_H, ctx.new_W
    align_corners = ctx.align_corners
    half_pixel = getattr(ctx, "half_pixel_centers", False)

    if align_corners:
        scale_y = (h - 1.0) / max(new_h - 1.0, 1.0) if new_h > 1 else 0.0
        y_coords = np_mod.arange(new_h, dtype=np_mod.float32) * scale_y
        scale_x = (w - 1.0) / max(new_w - 1.0, 1.0) if new_w > 1 else 0.0
        x_coords = np_mod.arange(new_w, dtype=np_mod.float32) * scale_x
    else:
        scale_y = float(h) / float(new_h)
        scale_x = float(w) / float(new_w)
        if half_pixel:
            y_coords = (np_mod.arange(new_h, dtype=np_mod.float32) + 0.5) * scale_y - 0.5
            x_coords = (np_mod.arange(new_w, dtype=np_mod.float32) + 0.5) * scale_x - 0.5
        else:
            y_coords = np_mod.arange(new_h, dtype=np_mod.float32) * scale_y
            x_coords = np_mod.arange(new_w, dtype=np_mod.float32) * scale_x

    return np_mod.meshgrid(y_coords, x_coords, indexing="ij")


def _apply_resize_batch(np_mod: Any, imgs: Any, out: object, coords: object, order: int) -> Any:
    """Apply the resize operation across a batch of images using interpolation.

    Args:
        np_mod: NumPy-compatible execution module.
        imgs: Input batched images array of shape (B, H, W, C).
        out: Pre-allocated output array of shape (B, new_H, new_W, C).
        coords: Tuple of (y_grid, x_grid) coordinate meshgrid.
        order (int): Spline interpolation order (0, 1, or 3).

    Returns:
        Any: Resized output array.
    """
    if imgs is None or coords is None:
        return 0

    np_mod = np if np_mod is None else np_mod
    b_count, _, _, c_count = imgs.shape
    y_coords, x_coords = coords  # type: ignore[misc]

    target_out = out if isinstance(out, np.ndarray) else np_mod.zeros((b_count, y_coords.shape[0], y_coords.shape[1], c_count), dtype=imgs.dtype)

    for b in range(b_count):
        for c in range(c_count):
            target_out[b, :, :, c] = _np_map_coordinates(
                np_mod,
                imgs[b, :, :, c],
                (y_coords, x_coords),
                order=order,
                fill_value=0.0,
            )
    return target_out


def _extract_upsample_size(
    spatial_shape: tuple[int, ...],
    size: Any,
    scale_factor: Any,
) -> tuple[int, ...]:
    """Calculate target spatial dimensions from size or scale_factor arguments.

    Args:
        spatial_shape: Current spatial dimensions of input tensor.
        size: Target output size integer or sequence.
        scale_factor: Multiplier float or sequence of multipliers.

    Returns:
        tuple[int, ...]: Resolved target spatial dimensions.
    """
    dim_count = len(spatial_shape)
    if size is not None:
        if isinstance(size, int):
            return (size,) * dim_count
        return tuple(size)

    if scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            return tuple(int(dim * scale_factor) for dim in spatial_shape)
        return tuple(int(dim * factor) for dim, factor in zip(spatial_shape, scale_factor))

    return spatial_shape


def _upsample_1d_linear(
    x: np.ndarray,
    target_len: int,
    align_corners: bool = False,
) -> np.ndarray:
    """Upsample 1D tensor along innermost dimension using linear interpolation.

    Args:
        x: Input array of shape (B, C, L).
        target_len: Target length for last dimension.
        align_corners: Whether grid extrema align with corner endpoints.

    Returns:
        np.ndarray: Upsampled 1D array of shape (B, C, target_len).
    """
    b, c, orig_len = x.shape
    if orig_len == target_len:
        return x.copy()

    if align_corners:
        scale = (orig_len - 1.0) / max(target_len - 1.0, 1.0) if target_len > 1 else 0.0
        src = np.arange(target_len, dtype=np.float32) * scale
    else:
        scale = float(orig_len) / float(target_len)
        src = (np.arange(target_len, dtype=np.float32) + 0.5) * scale - 0.5

    i0 = np.clip(np.floor(src).astype(int), 0, orig_len - 1)
    i1 = np.clip(i0 + 1, 0, orig_len - 1)
    w = np.clip(src - i0, 0.0, 1.0)

    out = np.zeros((b, c, target_len), dtype=x.dtype)
    for b_idx in range(b):
        for c_idx in range(c):
            out[b_idx, c_idx] = (1.0 - w) * x[b_idx, c_idx, i0] + w * x[b_idx, c_idx, i1]
    return out


def _upsample_3d_trilinear(
    x: np.ndarray,
    target_shape: tuple[int, int, int],
    align_corners: bool = False,
) -> np.ndarray:
    """Upsample 3D tensor across spatial dimensions using trilinear interpolation.

    Args:
        x: Input array of shape (B, C, D, H, W).
        target_shape: Target spatial dimensions (target_d, target_h, target_w).
        align_corners: Whether grid extrema align with corner voxels.

    Returns:
        np.ndarray: Upsampled 3D array of shape (B, C, target_d, target_h, target_w).
    """
    b, c, d, h, w = x.shape
    td, th, tw = target_shape

    def _get_coords(orig_len: int, target_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute coordinate indices and linear interpolation weights for a single dimension.

        Args:
            orig_len (int): Original dimension size.
            target_len (int): Target upsampled dimension size.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (idx0, idx1, weight) arrays.
        """
        if align_corners:
            scale = (orig_len - 1.0) / max(target_len - 1.0, 1.0) if target_len > 1 else 0.0
            src = np.arange(target_len, dtype=np.float32) * scale
        else:
            scale = float(orig_len) / float(target_len)
            src = (np.arange(target_len, dtype=np.float32) + 0.5) * scale - 0.5
        idx0 = np.clip(np.floor(src).astype(int), 0, orig_len - 1)
        idx1 = np.clip(idx0 + 1, 0, orig_len - 1)
        weight = np.clip(src - idx0, 0.0, 1.0)
        return idx0, idx1, weight

    d0, d1, wd = _get_coords(d, td)
    h0, h1, wh = _get_coords(h, th)
    w0, w1, ww = _get_coords(w, tw)

    out = np.zeros((b, c, td, th, tw), dtype=x.dtype)
    for b_idx in range(b):
        for c_idx in range(c):
            vol = x[b_idx, c_idx]
            for z in range(td):
                for y in range(th):
                    c000 = vol[d0[z], h0[y], w0]
                    c001 = vol[d0[z], h0[y], w1]
                    c010 = vol[d0[z], h1[y], w0]
                    c011 = vol[d0[z], h1[y], w1]
                    c100 = vol[d1[z], h0[y], w0]
                    c101 = vol[d1[z], h0[y], w1]
                    c110 = vol[d1[z], h1[y], w0]
                    c111 = vol[d1[z], h1[y], w1]

                    c00 = (1.0 - ww) * c000 + ww * c001
                    c01 = (1.0 - ww) * c010 + ww * c011
                    c10 = (1.0 - ww) * c100 + ww * c101
                    c11 = (1.0 - ww) * c110 + ww * c111

                    c0 = (1.0 - wh[y]) * c00 + wh[y] * c01
                    c1 = (1.0 - wh[y]) * c10 + wh[y] * c11

                    out[b_idx, c_idx, z, y] = (1.0 - wd[z]) * c0 + wd[z] * c1
    return out


@global_eager_registry.register("UpsampleNearest")
def _upsample_nearest_eager(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _upsample_nearest_eager operation across 1D, 2D, or 3D tensors.

    Args:
        backend_module: Execution backend module.
        *args: Positional arguments containing input tensor and optional size.
        **kwargs: Keyword arguments containing size, scale_factor, and data_format.

    Returns:
        Any: Upsampled tensor using nearest-neighbor interpolation.
    """
    if not args or args[0] is None:
        return 0

    x = args[0]
    np_mod = np if backend_module is None else getattr(backend_module, "numpy", np)
    name: str = getattr(backend_module, "__name__", "numpy") if backend_module is not None else "numpy"

    arr = np_mod.asarray(_to_numpy_array(np_mod, x, name))

    size = kwargs.get("size", args[1] if len(args) > 1 else None)
    scale_factor = kwargs.get("scale_factor", args[2] if len(args) > 2 else None)

    # Standard layout is (B, C, *spatial_dims)
    b, c = arr.shape[:2]
    spatial_shape = arr.shape[2:]
    target_spatial = _extract_upsample_size(spatial_shape, size, scale_factor)

    if len(spatial_shape) == 1:
        orig_len = spatial_shape[0]
        t_len = target_spatial[0]
        scale = float(orig_len) / float(t_len)
        src = (np.arange(t_len, dtype=np.float32) + 0.5) * scale - 0.5
        idx = np.clip(np.round(src).astype(int), 0, orig_len - 1)
        out = arr[:, :, idx]
    elif len(spatial_shape) == 2:
        # Channels-first to channels-last for 2D resize helper
        arr_last = np.transpose(arr, (0, 2, 3, 1))
        ctx = ResizeContext(
            H=spatial_shape[0],
            W=spatial_shape[1],
            new_H=target_spatial[0],
            new_W=target_spatial[1],
            align_corners=False,
            half_pixel_centers=True,
        )
        coords = _compute_resize_grid(np_mod, ctx)
        out_last = np_mod.zeros(
            (b, target_spatial[0], target_spatial[1], c),
            dtype=arr.dtype,
        )
        out_last = _apply_resize_batch(np_mod, arr_last, out_last, coords, order=0)
        out = np.transpose(out_last, (0, 3, 1, 2))
    elif len(spatial_shape) == 3:
        td, th, tw = target_spatial
        d, h, w = spatial_shape
        sd, sh, sw = float(d) / float(td), float(h) / float(th), float(w) / float(tw)
        id_z = np.clip(np.round((np.arange(td, dtype=np.float32) + 0.5) * sd - 0.5).astype(int), 0, d - 1)
        id_y = np.clip(np.round((np.arange(th, dtype=np.float32) + 0.5) * sh - 0.5).astype(int), 0, h - 1)
        id_x = np.clip(np.round((np.arange(tw, dtype=np.float32) + 0.5) * sw - 0.5).astype(int), 0, w - 1)
        out = arr[:, :, id_z[:, None, None], id_y[None, :, None], id_x[None, None, :]]
    else:
        out = arr.copy()

    return _from_numpy_array(backend_module, out, name, x)


@global_eager_registry.register("UpsampleBilinear")
@global_eager_registry.register("UpsampleTrilinear")
@global_eager_registry.register("UpsampleLinear")
def _upsample_linear_eager(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate linear, bilinear, or trilinear upsampling across spatial dimensions.

    Args:
        backend_module: Execution backend module.
        *args: Positional arguments containing input tensor and optional size.
        **kwargs: Keyword arguments containing size, scale_factor, and align_corners.

    Returns:
        Any: Linearly interpolated upsampled tensor.
    """
    if not args or args[0] is None:
        return 0

    x = args[0]
    np_mod = np if backend_module is None else getattr(backend_module, "numpy", np)
    name: str = getattr(backend_module, "__name__", "numpy") if backend_module is not None else "numpy"

    arr = np_mod.asarray(_to_numpy_array(np_mod, x, name))

    size = kwargs.get("size", args[1] if len(args) > 1 else None)
    scale_factor = kwargs.get("scale_factor", args[2] if len(args) > 2 else None)
    align_corners = bool(kwargs.get("align_corners", False))

    b, c = arr.shape[:2]
    spatial_shape = arr.shape[2:]
    target_spatial = _extract_upsample_size(spatial_shape, size, scale_factor)

    if len(spatial_shape) == 1:
        out = _upsample_1d_linear(arr, target_spatial[0], align_corners=align_corners)
    elif len(spatial_shape) == 2:
        arr_last = np.transpose(arr, (0, 2, 3, 1))
        ctx = ResizeContext(
            H=spatial_shape[0],
            W=spatial_shape[1],
            new_H=target_spatial[0],
            new_W=target_spatial[1],
            align_corners=align_corners,
            half_pixel_centers=not align_corners,
        )
        coords = _compute_resize_grid(np_mod, ctx)
        out_last = np_mod.zeros(
            (b, target_spatial[0], target_spatial[1], c),
            dtype=arr.dtype,
        )
        out_last = _apply_resize_batch(np_mod, arr_last, out_last, coords, order=1)
        out = np.transpose(out_last, (0, 3, 1, 2))
    elif len(spatial_shape) == 3:
        out = _upsample_3d_trilinear(
            arr,
            (target_spatial[0], target_spatial[1], target_spatial[2]),
            align_corners=align_corners,
        )
    else:
        out = arr.copy()

    return _from_numpy_array(backend_module, out, name, x)


@global_eager_registry.register("UpsampleBicubic")
def _upsample_bicubic_eager(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate 2D bicubic spatial upsampling using Catmull-Rom cubic splines.

    Args:
        backend_module: Execution backend module.
        *args: Positional arguments containing input tensor and optional size.
        **kwargs: Keyword arguments containing size, scale_factor, and align_corners.

    Returns:
        Any: Bicubically interpolated upsampled tensor.
    """
    if not args or args[0] is None:
        return 0

    x = args[0]
    np_mod = np if backend_module is None else getattr(backend_module, "numpy", np)
    name: str = getattr(backend_module, "__name__", "numpy") if backend_module is not None else "numpy"

    arr = np_mod.asarray(_to_numpy_array(np_mod, x, name))

    size = kwargs.get("size", args[1] if len(args) > 1 else None)
    scale_factor = kwargs.get("scale_factor", args[2] if len(args) > 2 else None)
    align_corners = bool(kwargs.get("align_corners", False))

    b, c = arr.shape[:2]
    spatial_shape = arr.shape[2:]
    target_spatial = _extract_upsample_size(spatial_shape, size, scale_factor)

    arr_last = np.transpose(arr, (0, 2, 3, 1))
    ctx = ResizeContext(
        H=spatial_shape[0],
        W=spatial_shape[1],
        new_H=target_spatial[0],
        new_W=target_spatial[1],
        align_corners=align_corners,
        half_pixel_centers=not align_corners,
    )
    coords = _compute_resize_grid(np_mod, ctx)
    out_last = np_mod.zeros((b, target_spatial[0], target_spatial[1], c), dtype=arr.dtype)
    out_last = _apply_resize_batch(np_mod, arr_last, out_last, coords, order=3)
    out = np.transpose(out_last, (0, 3, 1, 2))

    return _from_numpy_array(backend_module, out, name, x)


def resize_eager(
    backend_module: Any,
    images: Any,
    size: tuple[int, int],
    config: ResizeOptions,
) -> Any:
    """Evaluate resize_eager operation.

    Args:
        backend_module: Execution backend module.
        images: Input images tensor or array.
        size (tuple[int, int]): Desired (new_H, new_W) spatial output dimensions.
        config (ResizeOptions): Configuration options for interpolation and corner alignment.

    Returns:
        Any: Resized images matching input backend representation.
    """
    if images is None or size is None:
        return 0

    np_mod = np if backend_module is None else getattr(backend_module, "numpy", np)
    name: str = getattr(backend_module, "__name__", "numpy") if backend_module is not None else "numpy"

    arr = np_mod.asarray(_to_numpy_array(np_mod, images, name))

    data_format = getattr(config, "data_format", None)
    arr, was_3d = _standardize_image_input(arr, data_format)

    b, h, w, c = arr.shape
    new_h, new_w = size

    interp_name = getattr(config, "interpolation", "bilinear")
    order = _get_resize_interpolation_order(interp_name)
    align_corners = bool(getattr(config, "align_corners", False))
    half_pixel = bool(getattr(config, "half_pixel_centers", False))

    ctx = ResizeContext(
        H=h,
        W=w,
        new_H=new_h,
        new_W=new_w,
        align_corners=align_corners,
        half_pixel_centers=half_pixel,
    )
    coords = _compute_resize_grid(np_mod, ctx)
    out = np_mod.zeros((b, new_h, new_w, c), dtype=arr.dtype)
    out = _apply_resize_batch(np_mod, arr, out, coords, order=order)

    if data_format == "channels_first":
        out = np.transpose(out, (0, 3, 1, 2))
    if was_3d and out.shape[0] == 1:
        out = out[0]

    return _from_numpy_array(backend_module, out, name, images)
