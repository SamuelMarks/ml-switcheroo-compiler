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


def draw_bounding_boxes(
    images: Tensor,
    boxes: Tensor,
    colors: Tensor | None = None,
    texts: list[str] | None = None,
) -> Tensor:
    """Draw bounding boxes on a batch of images.

    Args:
        images (Tensor): The input images.
        boxes (Tensor): The bounding boxes.
        colors (Tensor | None): The colors for the boxes.
        texts (list[str] | None): The texts for the boxes.

    Returns:
    Tensor: The images with bounding boxes drawn.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        colors_data = colors.data if colors is not None else None
        data = backend.execute_op(
            "DrawBoundingBoxes", images.data, boxes.data, colors=colors_data, texts=texts
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "DrawBoundingBoxes",
        [images, boxes] + ([colors] if colors is not None else []),
        {"texts": texts},
        (),
        images.dtype,
    )


def crop_images(  # noqa: PLR0913
    images: Tensor,
    top_cropping: int,
    bottom_cropping: int,
    left_cropping: int,
    right_cropping: int,
    data_format: str | None = None,
) -> Tensor:
    """Crops images.

    Args:
        images: Input images.
        top_cropping: Top cropping.
        bottom_cropping: Bottom cropping.
        left_cropping: Left cropping.
        right_cropping: Right cropping.
        data_format: Data format.

    Returns:
        Tensor: Cropped images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "CropImages",
            images.data,
            top_cropping=top_cropping,
            bottom_cropping=bottom_cropping,
            left_cropping=left_cropping,
            right_cropping=right_cropping,
            data_format=data_format,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "CropImages",
        [images],
        {
            "top_cropping": top_cropping,
            "bottom_cropping": bottom_cropping,
            "left_cropping": left_cropping,
            "right_cropping": right_cropping,
            "data_format": data_format,
        },
        (),
        images.dtype,
    )


def extract_patches(  # noqa: PLR0913
    images: Tensor,
    size: int | tuple[int, int] | list[int],
    strides: int | tuple[int, int] | list[int] | None = None,
    dilation_rate: int | tuple[int, int] | list[int] | None = None,
    padding: str = "valid",
    data_format: str | None = None,
) -> Tensor:
    """Extracts patches from images.

    Args:
        images: Input images.
        size: Patch size.
        strides: Strides.
        dilation_rate: Dilation rate.
        padding: Padding.
        data_format: Data format.

    Returns:
        Tensor: Patches.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "ExtractPatches",
            images.data,
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            data_format=data_format,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "ExtractPatches",
        [images],
        {
            "size": size,
            "strides": strides,
            "dilation_rate": dilation_rate,
            "padding": padding,
            "data_format": data_format,
        },
        (),
        images.dtype,
    )


def pad_images(  # noqa: PLR0913
    images: Tensor,
    top_padding: int,
    bottom_padding: int,
    left_padding: int,
    right_padding: int,
    target_height: int | None = None,
    target_width: int | None = None,
    data_format: str | None = None,
) -> Tensor:
    """Pads images.

    Args:
        images: Input images.
        top_padding: Top padding.
        bottom_padding: Bottom padding.
        left_padding: Left padding.
        right_padding: Right padding.
        target_height: Target height.
        target_width: Target width.
        data_format: Data format.

    Returns:
        Tensor: Padded images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "PadImages",
            images.data,
            top_padding=top_padding,
            bottom_padding=bottom_padding,
            left_padding=left_padding,
            right_padding=right_padding,
            target_height=target_height,
            target_width=target_width,
            data_format=data_format,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "PadImages",
        [images],
        {
            "top_padding": top_padding,
            "bottom_padding": bottom_padding,
            "left_padding": left_padding,
            "right_padding": right_padding,
            "target_height": target_height,
            "target_width": target_width,
            "data_format": data_format,
        },
        (),
        images.dtype,
    )
