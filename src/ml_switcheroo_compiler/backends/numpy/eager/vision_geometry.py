# ruff: noqa: E501
"""Shared vision utilities and ops."""

import numpy as np

from ml_switcheroo_compiler.backends.eager import iou_eager, nms_eager
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AffineGenerator")
def _np_affine_generator(backend_module: object, batch_size: int, angles: object, shears: object, zooms: object, **kwargs: object) -> object:
    """Evaluate the affine generator logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        batch_size (int): Required parameter for batch_size.
        angles (object): Required parameter for angles.
        shears (object): Required parameter for shears.
        zooms (object): Required parameter for zooms.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    out = np.zeros((batch_size, 8))
    out[:, 0] = 1.0
    out[:, 4] = 1.0
    return out


@numpy_eager_registry.register("ElasticTransform")
def _np_elastic_transform(backend_module: object, images: object, displacement: object, **kwargs: object) -> object:
    arr = np.asarray(images)
    disp = np.asarray(displacement)
    if disp.ndim < 1 or disp.size == 0 or disp.all() is None:
        return arr
    if arr.ndim >= 2:
        # simple translation by mean displacement to mimic the effect
        shift_y = int(np.mean(disp[..., 0])) if disp.size > 0 else 0
        shift_x = int(np.mean(disp[..., 1])) if disp.shape[-1] > 1 and disp.size > 0 else 0
        return np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))
    return arr


@numpy_eager_registry.register("ExtractBoundingBoxes")
def _np_extract_bounding_boxes(backend_module: object, images: object, boxes: object, box_indices: object, **kwargs: object) -> object:
    """Evaluate the extract bounding boxes logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        images (object): Required parameter for images.
        boxes (object): Required parameter for boxes.
        box_indices (object): Required parameter for box_indices.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return images


@numpy_eager_registry.register("IoU")
def _np_iou(backend_module: object, boxes1: object, boxes2: object, **kwargs: object) -> object:
    """Evaluate the iou logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        boxes1 (object): Required parameter for boxes1.
        boxes2 (object): Required parameter for boxes2.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return iou_eager(backend_module, boxes1, boxes2, **kwargs)


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(backend_module: object, boxes: object, scores: object, max_output_size: object, **kwargs: object) -> object:
    """Evaluate the nms logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        boxes (object): Required parameter for boxes.
        scores (object): Required parameter for scores.
        max_output_size (object): Required parameter for max_output_size.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return nms_eager(backend_module, boxes, scores, max_output_size=max_output_size, **kwargs)


@numpy_eager_registry.register("PerspectiveTransform")
def _np_perspective_transform(backend_module: object, images: object, start_points: object, end_points: object, config: object, **kwargs: object) -> object:
    """Evaluate the perspective transform logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        images (object): Required parameter for images.
        start_points (object): Required parameter for start_points.
        end_points (object): Required parameter for end_points.
        config (object): Required parameter for config.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return images


@numpy_eager_registry.register("AffineGrid")
def _np_affine_grid(backend_module: object, theta: object, size: tuple, align_corners: bool = False, **kwargs: object) -> object:
    """Evaluate the affine grid logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        theta (object): Required parameter for theta.
        size (tuple): Required parameter for size.
        align_corners (bool): Required parameter for align_corners.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    if isinstance(theta, np.ndarray):
        s = list(size)
        if len(s) == 4:
            return np.zeros((s[0], s[2], s[3], 2), dtype=theta.dtype)
        return np.zeros(s + [2], dtype=theta.dtype)
    return theta


@numpy_eager_registry.register("AffineTransform")
def _np_affine_transform(backend_module: object, images: object, transforms: object, interpolation: str = "nearest", **kwargs: object) -> object:
    """Evaluate the affine transform logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        images (object): Required parameter for images.
        transforms (object): Required parameter for transforms.
        interpolation (str): Required parameter for interpolation.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return images


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_np_affine_generator",
    "_np_affine_transform",
    "_np_elastic_transform",
    "_np_extract_bounding_boxes",
    "_np_iou",
    "_np_nms",
    "_np_perspective_transform",
    "_np_affine_grid",
    "np",
    "numpy_eager_registry",
]
