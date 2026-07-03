"""Vision operations."""

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
        padding (tuple[int, int, int, int]): The padding.
        target_shape (tuple[int, int]): The target shape.
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
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
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
    return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))


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
        boxes (Tensor): Bounding boxes.
        **kwargs: Extra arguments.
        strides (int | tuple[int, int] | list[int] | None): Strides.
        dilation_rate (int | tuple[int, int] | list[int] | None): Dilation rate.
        padding (str): Padding.
        data_format (str | None): Data format.
        boxes (Tensor): Bounding boxes [num_boxes, 4] with coords [y1, x1, y2, x2].
        box_indices (Tensor): 1-D tensor of size [num_boxes] with indices to images.
        config_obj (BBoxConfig | None): Configuration.

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
        padding (tuple[int, int, int, int]): The padding.
        target_shape (tuple[int, int]): The target shape.
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
        padding (tuple[int, int, int, int]): The padding.
        target_shape (tuple[int, int]): The target shape.
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
        padding (tuple[int, int, int, int]): The padding.
        target_shape (tuple[int, int]): The target shape.
        boxes (Tensor): The bounding boxes.
        **kwargs: Extra arguments.
        strides (int | tuple[int, int] | list[int] | None): Strides.
        dilation_rate (int | tuple[int, int] | list[int] | None): Dilation rate.
        padding (str): Padding.
        data_format (str | None): Data format.
        colors (Tensor | None): The colors for the boxes.
        texts (list[str] | None): The texts for the boxes.

    Returns:
    Tensor: The images with bounding boxes drawn.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        colors_data = colors.data if colors is not None else None
        data = backend.execute_op("DrawBoundingBoxes", images.data, boxes.data, colors=colors_data, texts=texts)
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


def crop_images(
    images: Tensor,
    cropping: tuple[int, int, int, int],
    data_format: str | None = None,
) -> Tensor:
    """Crops images.

    Args:
        images (Tensor): Input images.
        cropping (tuple[int, int, int, int]): Cropping.
        data_format (str | None): Data format.

    Returns:
        Tensor: Cropped images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "CropImages",
            images.data,
            top_cropping=cropping[0],
            bottom_cropping=cropping[1],
            left_cropping=cropping[2],
            right_cropping=cropping[3],
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
            "top_cropping": cropping[0],
            "bottom_cropping": cropping[1],
            "left_cropping": cropping[2],
            "right_cropping": cropping[3],
            "data_format": data_format,
        },
        (),
        images.dtype,
    )


def extract_patches(
    images: Tensor,
    size: int | tuple[int, int] | list[int],
    strides: int | tuple[int, int] | list[int] | None = None,
    dilation_rate: int | tuple[int, int] | list[int] | None = None,
    padding: str = "valid",
    data_format: str | None = None,
    **kwargs: object,
) -> Tensor:
    """Extracts patches from images.

    Args:
        images (Tensor): Input images.
        size (int | tuple[int, int] | list[int]): Patch size.
        **kwargs: Extra arguments.
        strides (int | tuple[int, int] | list[int] | None): Strides.
        dilation_rate (int | tuple[int, int] | list[int] | None): Dilation rate.
        padding (str): Padding.
        data_format (str | None): Data format.

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


def pad_images(
    images: Tensor,
    padding: tuple[int, int, int, int],
    target_shape: tuple[int | None, int | None],
    data_format: str | None = None,
) -> Tensor:
    """Pads images.

    Args:
        images (Tensor): Input images.
        padding (tuple[int, int, int, int]): Padding.
        target_shape (tuple[int, int]): Target shape.
        data_format (str | None): Data format.

    Returns:
        Tensor: Padded images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "PadImages",
            images.data,
            top_padding=padding[0],
            bottom_padding=padding[1],
            left_padding=padding[2],
            right_padding=padding[3],
            target_height=target_shape[0],
            target_width=target_shape[1],
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
            "top_padding": padding[0],
            "bottom_padding": padding[1],
            "left_padding": padding[2],
            "right_padding": padding[3],
            "target_height": target_shape[0],
            "target_width": target_shape[1],
            "data_format": data_format,
        },
        (),
        images.dtype,
    )
