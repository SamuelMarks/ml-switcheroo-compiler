"""Module bbox.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Vision operations."""
from dataclasses import dataclass
from typing import Any

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@dataclass
class ExtractPatchesOptions:
    """Options for extracting patches."""

    strides: int | tuple[int, int] | list[int] | None = None
    dilation_rate: int | tuple[int, int] | list[int] | None = None
    padding: str = "valid"
    data_format: str | None = None


def crop_and_resize(
    images: Tensor,  # type: ignore
    boxes: Tensor,  # type: ignore
    box_indices: Tensor,  # type: ignore
    crop_size: tuple[int, int],
    **kwargs: Any,
) -> Any:
    """Extract crops from the input image tensor and resizes them.

    Args:
        images (Tensor): The images parameter.
        boxes (Tensor): The boxes parameter.
        box_indices (Tensor): The box_indices parameter.
        crop_size (tuple): The crop_size parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
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
    images: Tensor,  # type: ignore
    boxes: Tensor,  # type: ignore
    box_indices: Tensor,  # type: ignore
    config_obj: Any,
) -> Any:
    """Evaluate _extract_bounding_boxes_eager operation.

    Args:
        images (Tensor): The images parameter.
        boxes (Tensor): The boxes parameter.
        box_indices (Tensor): The box_indices parameter.
        config_obj (object): The config_obj parameter.

    Returns:
        Tensor: Result.
    """
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
    images: Tensor,  # type: ignore
    boxes: Tensor,  # type: ignore
    box_indices: Tensor,  # type: ignore
    config_obj: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Extract crops from the input image tensor and resizes them.

    Args:
        images (Tensor): The images parameter.
        boxes (Tensor): The boxes parameter.
        box_indices (Tensor): The box_indices parameter.
        config_obj (object): The config_obj parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import BBoxConfig

        crop_size = kwargs.get("crop_size", (0, 0))
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        config_obj = BBoxConfig(
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
    images: Tensor,  # type: ignore
    offset_height: int,
    offset_width: int,
    target_height: int,
    target_width: int,
) -> Any:
    """Crops an image to a specified bounding box.

    Args:
        images (Tensor): The images parameter.
        offset_height (int): The offset_height parameter.
        offset_width (int): The offset_width parameter.
        target_height (int): The target_height parameter.
        target_width (int): The target_width parameter.

    Returns:
        Tensor: Result.
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
    images: Tensor,  # type: ignore
    offset_height: int,
    offset_width: int,
    target_height: int,
    target_width: int,
) -> Any:
    """Pad an image with zeros to the specified height and width.

    Args:
        images (Tensor): The images parameter.
        offset_height (int): The offset_height parameter.
        offset_width (int): The offset_width parameter.
        target_height (int): The target_height parameter.
        target_width (int): The target_width parameter.

    Returns:
        Tensor: Result.
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
    images: Tensor,  # type: ignore
    boxes: Tensor,  # type: ignore
    colors: Tensor | None = None,  # type: ignore
    texts: list[str] | None = None,
) -> Any:
    """Draw bounding boxes on a batch of images.

    Args:
        images (Tensor): The images parameter.
        boxes (Tensor): The boxes parameter.
        colors (object): The colors parameter.
        texts (object): The texts parameter.

    Returns:
        Tensor: Result.
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
    images: Tensor,  # type: ignore
    cropping: tuple[int, int, int, int],
    data_format: str | None = None,
) -> Any:
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
    images: Tensor,  # type: ignore
    size: int | tuple[int, int] | list[int],
    options: ExtractPatchesOptions | None = None,
    **kwargs: Any,
) -> Any:
    """Extract patches from images.

    Args:
        images (Tensor): Input images.
        size (int | tuple[int, int] | list[int]): Patch size.
        options (ExtractPatchesOptions): Options.
        **kwargs: Extra arguments.

    Returns:
        Tensor: Patches.
    """
    options = options or ExtractPatchesOptions()
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "ExtractPatches",
            images.data,
            size=size,
            strides=options.strides,
            dilation_rate=options.dilation_rate,
            padding=options.padding,
            data_format=options.data_format,
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
            "strides": options.strides,
            "dilation_rate": options.dilation_rate,
            "padding": options.padding,
            "data_format": options.data_format,
        },
        (),
        images.dtype,
    )


def pad_images(
    images: Tensor,  # type: ignore
    padding: tuple[int, int, int, int],
    target_shape: tuple[int | None, int | None],
    data_format: str | None = None,
) -> Any:
    """Pad images.

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


@register_op("ExtractBoundingBoxes")
class ExtractBoundingBoxes(OpDef):
    """ExtractBoundingBoxes operation."""

    op_name = "ExtractBoundingBoxes"

    def infer_shape(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
        inputs (object): The inputs parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("Iou")
class Iou(OpDef):
    """Iou operation."""

    op_name = "Iou"

    def infer_shape(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(inputs, "shape", ())


@register_op("Nms")
class Nms(OpDef):
    """Nms operation."""

    op_name = "Nms"

    def infer_shape(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            inputs (object): The inputs parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(inputs, "shape", ())
