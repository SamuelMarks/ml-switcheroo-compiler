"""Vision operations."""

from __future__ import annotations

from __future__ import annotations
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def crop_and_resize(
    images: Tensor,
    boxes: Tensor,
    box_indices: Tensor,
    crop_size: tuple[int, int],
    **kwargs: object,
) -> Tensor:
    """Extracts crops from the input image tensor and resizes them.

    Args:
        images (Tensor): The input images.
        boxes (Tensor): Bounding boxes.
        box_indices (Tensor): Box indices.
        crop_size (tuple[int, int]): The crop size.
        **kwargs (object): Interpolation method, extrapolation_value.

    Returns:
        Tensor: The cropped and resized images.
    """
    method = kwargs.get("method", "bilinear")
    extrapolation_value = kwargs.get("extrapolation_value", 0.0)

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "CropAndResize",
            images.data,
            boxes.data,
            box_indices.data,
            crop_size=crop_size,
            method=method,
            extrapolation_value=extrapolation_value,
        )
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
    return _emit_shape_node(
        "CropAndResize",
        [images, boxes, box_indices],
        {"crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value},
        (),
        DType.Int32,
    )


def _extract_bounding_boxes_eager(
    images: Tensor,
    boxes: Tensor,
    box_indices: Tensor,
    config_obj: object,
) -> Tensor:
    """Evaluate extract_bounding_boxes eagerly."""
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    data = backend.execute_op(
        "ExtractBoundingBoxes",
        images.data,
        boxes.data,
        box_indices.data,
        config=config_obj,
    )
    return Tensor(
        backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
    )


def extract_bounding_boxes(
    images: Tensor,
    boxes: Tensor,
    box_indices: Tensor,
    config_obj: object | None = None,
    **kwargs: object,
) -> Tensor:
    """Extracts crops from the input image tensor and resizes them.

    Args:
        images (Tensor): Input images.
        boxes (Tensor): Bounding boxes [num_boxes, 4] with coords [y1, x1, y2, x2].
        box_indices (Tensor): 1-D tensor of size [num_boxes] with indices to images.
        config_obj (BBoxConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: Cropped and resized images of shape [num_boxes, crop_height, crop_width, C].
    """
    if config_obj is None:  # pragma: no branch
        from ml_switcheroo_compiler.ops.configs import BBoxConfig  # pragma: no cover

        crop_size = kwargs.get("crop_size", (0, 0))  # pragma: no cover
        if isinstance(crop_size, int):  # pragma: no cover
            crop_size = (crop_size, crop_size)  # pragma: no cover
        config_obj = BBoxConfig(  # pragma: no cover
            crop_size=crop_size,
            interpolation=kwargs.get("interpolation", "bilinear"),
            extrapolation_value=kwargs.get("extrapolation_value", 0.0),
            data_format=kwargs.get("data_format", None),
        )

    if config.eager_mode:
        return _extract_bounding_boxes_eager(images, boxes, box_indices, config_obj)

    return _emit_shape_node(
        "ExtractBoundingBoxes",
        [images, boxes, box_indices],
        {
            "config": config_obj,
        },
        (),
        DType.Int32,
    )


def crop(
    images: Tensor,
    offset_height: int,
    offset_width: int,
    target_height: int,
    target_width: int,
) -> Tensor:
    """Crops an image to a specified bounding box.

    Args:
        images (Tensor): The input images.
        offset_height (int): Vertical coordinate of the top-left corner.
        offset_width (int): Horizontal coordinate of the top-left corner.
        target_height (int): Height of the result.
        target_width (int): Width of the result.

    Returns:
        Tensor: The cropped images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "Crop",
            images.data,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=target_height,
            target_width=target_width,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "Crop",
        [images],
        {
            "offset_height": offset_height,
            "offset_width": offset_width,
            "target_height": target_height,
            "target_width": target_width,
        },
        (),
        images.dtype,
    )


def pad_to_bounding_box(
    images: Tensor,
    offset_height: int,
    offset_width: int,
    target_height: int,
    target_width: int,
) -> Tensor:
    """Pads an image with zeros to the specified height and width.

    Args:
        images (Tensor): The input images.
        offset_height (int): Number of rows of zeros to add on top.
        offset_width (int): Number of columns of zeros to add on the left.
        target_height (int): Height of the result.
        target_width (int): Width of the result.

    Returns:
        Tensor: The padded images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "PadToBoundingBox",
            images.data,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=target_height,
            target_width=target_width,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "PadToBoundingBox",
        [images],
        {
            "offset_height": offset_height,
            "offset_width": offset_width,
            "target_height": target_height,
            "target_width": target_width,
        },
        (),
        images.dtype,
    )
