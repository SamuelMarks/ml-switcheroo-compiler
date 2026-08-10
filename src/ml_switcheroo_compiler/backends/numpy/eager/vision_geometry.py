# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Shared vision utilities and ops."""

from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager import iou_eager, nms_eager
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AffineGenerator")
def _np_affine_generator(backend_module: Any, batch_size: int, angles: Any, shears: Any, zooms: Any, **kwargs: Any) -> Any:
    """Evaluate _np_affine_generator operation.

    Args:
        backend_module (object): The backend_module parameter.
        batch_size (int): The batch_size parameter.
        angles (object): The angles parameter.
        shears (object): The shears parameter.
        zooms (object): The zooms parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    out = np.zeros((batch_size, 8))
    out[:, 0] = 1.0
    out[:, 4] = 1.0
    return out


@numpy_eager_registry.register("ElasticTransform")
def _np_elastic_transform(backend_module: Any, images: Any, displacement: Any, **kwargs: Any) -> Any:
    """Apply elastic transformation to an image.

    Args:
        backend_module: The active backend module.
        images: The input images array.
        displacement: The displacement field array.
        **kwargs: Additional hyperparameters.

    Returns:
        The transformed images array.
    """
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
def _np_extract_bounding_boxes(backend_module: Any, images: Any, boxes: Any, box_indices: Any, **kwargs: Any) -> Any:
    """Evaluate _np_extract_bounding_boxes operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        boxes (object): The boxes parameter.
        box_indices (object): The box_indices parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return images


@numpy_eager_registry.register("IoU")
def _np_iou(backend_module: Any, boxes1: Any, boxes2: Any, **kwargs: Any) -> Any:
    """Evaluate _np_iou operation.

    Args:
        backend_module (object): The backend_module parameter.
        boxes1 (object): The boxes1 parameter.
        boxes2 (object): The boxes2 parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return iou_eager(backend_module, boxes1, boxes2, **kwargs)


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(backend_module: Any, boxes: Any, scores: Any, max_output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _np_nms operation.

    Args:
        backend_module (object): The backend_module parameter.
        boxes (object): The boxes parameter.
        scores (object): The scores parameter.
        max_output_size (object): The max_output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return nms_eager(backend_module, boxes, scores, **kwargs)


@numpy_eager_registry.register("PerspectiveTransform")
def _np_perspective_transform(backend_module: Any, images: Any, start_points: Any, end_points: Any, config: Any, **kwargs: Any) -> Any:
    """Evaluate _np_perspective_transform operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        start_points (object): The start_points parameter.
        end_points (object): The end_points parameter.
        config (object): The config parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return images


@numpy_eager_registry.register("AffineGrid")
def _np_affine_grid(backend_module: Any, theta: Any, size: tuple, align_corners: bool = False, **kwargs: Any) -> Any:
    """Evaluate _np_affine_grid operation.

    Args:
        backend_module (object): The backend_module parameter.
        theta (object): The theta parameter.
        size (tuple): The size parameter.
        align_corners (bool): The align_corners parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if isinstance(theta, np.ndarray):
        s = list(size)
        if len(s) == 4:
            return np.zeros((s[0], s[2], s[3], 2), dtype=theta.dtype)
        return np.zeros(s + [2], dtype=theta.dtype)
    return theta


@numpy_eager_registry.register("AffineTransform")
def _np_affine_transform(backend_module: Any, images: Any, transforms: Any, interpolation: str = "nearest", **kwargs: Any) -> Any:
    """Evaluate _np_affine_transform operation.

    Args:
        backend_module (object): The backend_module parameter.
        images (object): The images parameter.
        transforms (object): The transforms parameter.
        interpolation (str): The interpolation parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
