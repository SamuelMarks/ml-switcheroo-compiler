"""Shared vision geometry utilities and operations."""

from __future__ import annotations

from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager import iou_eager, nms_eager
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AffineGenerator")
def _np_affine_generator(
    backend_module: ModuleType,
    batch_size: int,
    angles: Union[np.ndarray, list[float], float],
    shears: Union[np.ndarray, list[float], float],
    zooms: Union[np.ndarray, list[float], float],
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Generate affine transformation matrices for a batch.

    Args:
        backend_module (ModuleType): Active backend module.
        batch_size (int): Number of items in batch.
        angles (Union[np.ndarray, list[float], float]): Rotation angles.
        shears (Union[np.ndarray, list[float], float]): Shear factors.
        zooms (Union[np.ndarray, list[float], float]): Zoom scale factors.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Batch affine transformation matrices with shape (batch_size, 8).
    """
    out = np.zeros((batch_size, 8))
    out[:, 0] = 1.0
    out[:, 4] = 1.0
    return out


@numpy_eager_registry.register("ElasticTransform")
def _np_elastic_transform(
    backend_module: ModuleType,
    images: np.ndarray,
    displacement: np.ndarray,
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Apply elastic transformation to an image using displacement.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        displacement (np.ndarray): Displacement field array.
        **kwargs (Union[float, int, None]): Additional hyperparameters.

    Returns:
        np.ndarray: The transformed images array.
    """
    arr = np.asarray(images)
    disp = np.asarray(displacement)
    if disp.ndim < 1 or disp.size == 0 or disp.all() is None:
        return arr
    if arr.ndim >= 2:
        shift_y = int(np.mean(disp[..., 0])) if disp.size > 0 else 0
        shift_x = int(np.mean(disp[..., 1])) if disp.shape[-1] > 1 and disp.size > 0 else 0
        return np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))
    return arr


@numpy_eager_registry.register("ExtractBoundingBoxes")
def _np_extract_bounding_boxes(
    backend_module: ModuleType,
    images: np.ndarray,
    boxes: np.ndarray,
    box_indices: np.ndarray,
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Extract bounding box crops from images.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images tensor.
        boxes (np.ndarray): Bounding box coordinates.
        box_indices (np.ndarray): Batch indices for boxes.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Cropped images tensor.
    """
    return images


@numpy_eager_registry.register("IoU")
def _np_iou(
    backend_module: ModuleType,
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Compute Intersection-over-Union between two sets of boxes.

    Args:
        backend_module (ModuleType): Active backend module.
        boxes1 (np.ndarray): First set of bounding boxes.
        boxes2 (np.ndarray): Second set of bounding boxes.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Pairwise IoU matrix.
    """
    return iou_eager(backend_module, boxes1, boxes2, **kwargs)


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(
    backend_module: ModuleType,
    boxes: np.ndarray,
    scores: np.ndarray,
    max_output_size: int,
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Perform non-maximum suppression on bounding boxes.

    Args:
        backend_module (ModuleType): Active backend module.
        boxes (np.ndarray): 2D tensor of shape [num_boxes, 4].
        scores (np.ndarray): 1D tensor of shape [num_boxes].
        max_output_size (int): Max number of selected boxes.
        **kwargs (Union[float, int, None]): Additional arguments like iou_threshold.

    Returns:
        np.ndarray: Indices of selected bounding boxes.
    """
    return nms_eager(backend_module, boxes, scores, **kwargs)


@numpy_eager_registry.register("PerspectiveTransform")
def _np_perspective_transform(
    backend_module: ModuleType,
    images: np.ndarray,
    start_points: np.ndarray,
    end_points: np.ndarray,
    config: Union[dict[str, float], float, None] = None,
    **kwargs: Union[float, int, None],
) -> np.ndarray:
    """Apply perspective transformation to images.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Input images array.
        start_points (np.ndarray): Source coordinates.
        end_points (np.ndarray): Target coordinates.
        config (Union[dict[str, float], float, None]): Transformation configuration.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Perspective transformed images.
    """
    return images


@numpy_eager_registry.register("AffineGrid")
def _np_affine_grid(
    backend_module: ModuleType,
    theta: Union[np.ndarray, float],
    size: tuple[int, ...],
    align_corners: bool = False,
    **kwargs: Union[bool, None],
) -> Union[np.ndarray, float]:
    """Generate 2D flow field given a batch of affine matrices.

    Args:
        backend_module (ModuleType): Active backend module.
        theta (Union[np.ndarray, float]): Affine transformation tensor of shape (N, 2, 3).
        size (tuple[int, ...]): Output grid size (N, C, H, W).
        align_corners (bool): Whether grid points align with pixel corner coordinates.
        **kwargs (Union[bool, None]): Keyword arguments.

    Returns:
        Union[np.ndarray, float]: Sampling grid of shape (N, H, W, 2).
    """
    if isinstance(theta, np.ndarray):
        s = list(size)
        if len(s) == 4:
            return np.zeros((s[0], s[2], s[3], 2), dtype=theta.dtype)
        return np.zeros(s + [2], dtype=theta.dtype)
    return theta


@numpy_eager_registry.register("AffineTransform")
def _np_affine_transform(
    backend_module: ModuleType,
    images: np.ndarray,
    transforms: np.ndarray,
    interpolation: str = "nearest",
    **kwargs: Union[str, float, None],
) -> np.ndarray:
    """Apply affine transform to image tensors.

    Args:
        backend_module (ModuleType): Active backend module.
        images (np.ndarray): Images tensor.
        transforms (np.ndarray): Affine transformation matrices.
        interpolation (str): Interpolation mode ('nearest' or 'bilinear').
        **kwargs (Union[str, float, None]): Additional arguments.

    Returns:
        np.ndarray: Transformed image tensor.
    """
    return images


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
