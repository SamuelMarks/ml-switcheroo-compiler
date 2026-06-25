# ruff: noqa: F405, F403
"""Shared vision utilities and ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
import numpy as np


@numpy_eager_registry.register("AffineGenerator")
def _np_affine_generator(
    backend_module: object,
    batch_size: int,
    angles: object,
    shears: object,
    zooms: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        batch_size: Arg.
        angles: Arg.
        shears: Arg.
        zooms: Arg.
        kwargs: Arg.
    """
    return np.zeros((batch_size, 8))  # pragma: no cover


@numpy_eager_registry.register("AffineTransform")
def _np_affine_transform(
    backend_module: object, images: object, transforms: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        transforms: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("ElasticTransform")
def _np_elastic_transform(
    backend_module: object, images: object, displacement: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        displacement: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("ExtractBoundingBoxes")
def _np_extract_bounding_boxes(
    backend_module: object, images: object, boxes: object, box_indices: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        boxes: Arg.
        box_indices: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("IoU")
def _np_iou(
    backend_module: object,
    boxes1: object,
    boxes2: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        boxes1: Arg.
        boxes2: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.eager import iou_eager

    return iou_eager(backend_module, boxes1, boxes2, **kwargs)


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(
    backend_module: object,
    boxes: object,
    scores: object,
    max_output_size: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        boxes: Arg.
        scores: Arg.
        max_output_size: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.eager import nms_eager

    return nms_eager(backend_module, boxes, scores, max_output_size=max_output_size, **kwargs)


@numpy_eager_registry.register("PerspectiveTransform")
def _np_perspective_transform(
    backend_module: object,
    images: object,
    start_points: object,
    end_points: object,
    config: object,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        start_points: Arg.
        end_points: Arg.
        config: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("ResizeBicubic")
def _np_resize_bicubic(
    backend_module: object, images: object, size: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        size: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("ResizeLanczos3")
def _np_resize_lanczos3(
    backend_module: object, images: object, size: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        size: Arg.
        kwargs: Arg.
    """
    return images


@numpy_eager_registry.register("ResizeNearest")
def _np_resize_nearest(
    backend_module: object, images: object, size: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        images: Arg.
        size: Arg.
        kwargs: Arg.
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
    "_np_resize_bicubic",
    "_np_resize_lanczos3",
    "_np_resize_nearest",
    "np",
    "numpy_eager_registry",
]
