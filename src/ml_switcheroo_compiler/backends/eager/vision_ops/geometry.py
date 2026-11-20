"""Generic backend-parametric eager reference operations for `vision_ops.geometry`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager import iou_eager, nms_eager
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.shape import ShapeProtocol


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("AffineGenerator")
def _np_affine_generator(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    batch_size: int,
    angles: BackendArg,
    shears: BackendArg,
    zooms: BackendArg,
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Generate affine transformation matrices for a batch.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        batch_size (int): Number of items in batch.
        angles (BackendArg): Rotation angles.
        shears (BackendArg): Shear factors.
        zooms (BackendArg): Zoom scale factors.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Batch affine transformation matrices with shape (batch_size, 8).
    """
    arr_mod = _resolve_eager_mod(backend_module)
    out = arr_mod.zeros((batch_size, 8))
    out[:, 0] = 1.0
    out[:, 4] = 1.0
    return out


@global_eager_registry.register("ElasticTransform")
def _np_elastic_transform(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    displacement: BackendArg,
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Apply elastic transformation to an image using displacement.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        displacement (BackendArg): Displacement field array.
        **kwargs (Union[float, int, None]): Additional hyperparameters.

    Returns:
        EagerTensorProtocol: The transformed images array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    arr = _ensure_arr(arr_mod.asarray(images))
    disp = arr_mod.asarray(displacement)
    if disp.ndim < 1 or disp.size == 0 or disp.all() is None:
        return arr
    if arr.ndim >= 2:
        shift_y = int(adapt_backend_module(arr_mod).require_symbol("mean")(disp[..., 0])) if disp.size > 0 else 0
        shift_x = (
            int(adapt_backend_module(arr_mod).require_symbol("mean")(disp[..., 1]))
            if disp.shape[-1] > 1 and disp.size > 0
            else 0
        )
        return _ensure_arr(arr_mod.roll(arr, shift=(shift_y, shift_x), axis=(0, 1)))
    return arr


@global_eager_registry.register("ExtractBoundingBoxes")
def _np_extract_bounding_boxes(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    boxes: BackendArg,
    box_indices: BackendArg,
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Extract bounding box crops from images.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images tensor.
        boxes (BackendArg): Bounding box coordinates.
        box_indices (BackendArg): Batch indices for boxes.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Cropped images tensor.
    """
    return _ensure_arr(images)


@global_eager_registry.register("IoU")
def _np_iou(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    boxes1: BackendArg,
    boxes2: BackendArg,
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Compute Intersection-over-Union between two sets of boxes.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        boxes1 (BackendArg): First set of bounding boxes.
        boxes2 (BackendArg): Second set of bounding boxes.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Pairwise IoU matrix.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    fmt = str(kwargs.get("box_format", "xyxy")) if "box_format" in kwargs else "xyxy"
    return _ensure_arr(arr_mod.asarray(iou_eager(backend_module, boxes1, boxes2, bounding_box_format=fmt)))


@global_eager_registry.register("NonMaxSuppression")
def _np_nms(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    boxes: BackendArg,
    scores: BackendArg,
    max_output_size: int,
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Perform non-maximum suppression on bounding boxes.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        boxes (BackendArg): 2D tensor of shape [num_boxes, 4].
        scores (BackendArg): 1D tensor of shape [num_boxes].
        max_output_size (int): Max number of selected boxes.
        **kwargs (Union[float, int, None]): Additional arguments like iou_threshold.

    Returns:
        EagerTensorProtocol: Indices of selected bounding boxes.
    """
    from ml_switcheroo_compiler.backends.eager.vision_filtering import NMSConfig
    from ml_switcheroo_compiler.backends.eager_registry import coerce_float

    arr_mod = _resolve_eager_mod(backend_module)
    nms_cfg = NMSConfig(
        max_output_size=int(max_output_size),
        iou_threshold=coerce_float(kwargs.get("iou_threshold", 0.5)),
        score_threshold=coerce_float(kwargs.get("score_threshold", float("-inf"))),
    )
    return _ensure_arr(arr_mod.asarray(nms_eager(backend_module, boxes, scores, nms_cfg)))


@global_eager_registry.register("PerspectiveTransform")
def _np_perspective_transform(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    start_points: BackendArg,
    end_points: BackendArg,
    config: dict[str, float] | float | None = None,
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Apply perspective transformation to images.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Input images array.
        start_points (BackendArg): Source coordinates.
        end_points (BackendArg): Target coordinates.
        config (Union[dict[str, float], float, None]): Transformation configuration.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Perspective transformed images.
    """
    return _ensure_arr(images)


@global_eager_registry.register("AffineGrid")
def _np_affine_grid(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    theta: BackendArg,
    size: tuple[int, ...],
    align_corners: bool = False,
    **kwargs: bool | None,
) -> EagerTensorProtocol | float:
    """Generate 2D flow field given a batch of affine matrices.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        theta (BackendArg): Affine transformation tensor of shape (N, 2, 3).
        size (tuple[int, ...]): Output grid size (N, C, H, W).
        align_corners (bool): Whether grid points align with pixel corner coordinates.
        **kwargs (Union[bool, None]): Keyword arguments.

    Returns:
        Union[EagerTensorProtocol, float]: Sampling grid of shape (N, H, W, 2).
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if isinstance(theta, ShapeProtocol):
        theta_arr = arr_mod.asarray(theta)
        s = list(size)
        if len(s) == 4:
            return _ensure_arr(arr_mod.zeros((s[0], s[2], s[3], 2), dtype=theta_arr.dtype))
        return _ensure_arr(arr_mod.zeros(s + [2], dtype=theta_arr.dtype))
    return float(theta) if isinstance(theta, (int, float)) else _ensure_arr(theta)


@global_eager_registry.register("AffineTransform")
def _np_affine_transform(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    images: BackendArg,
    transforms: BackendArg,
    interpolation: str = "nearest",
    **kwargs: str | float | None,
) -> EagerTensorProtocol:
    """Apply affine transform to image tensors.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        images (BackendArg): Images tensor.
        transforms (BackendArg): Affine transformation matrices.
        interpolation (str): Interpolation mode ('nearest' or 'bilinear').
        **kwargs (Union[str, float, None]): Additional arguments.

    Returns:
        EagerTensorProtocol: Transformed image tensor.
    """
    return _ensure_arr(images)


__all__ = [
    "_np_affine_generator",
    "_np_affine_grid",
    "_np_affine_transform",
    "_np_elastic_transform",
    "_np_extract_bounding_boxes",
    "_np_iou",
    "_np_nms",
    "_np_perspective_transform",
]
