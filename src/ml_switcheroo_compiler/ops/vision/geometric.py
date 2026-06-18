"""Vision and Image processing operations."""

import typing

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def resize_bilinear(images: Tensor, size: tuple[int, int], align_corners: bool = False) -> Tensor:
    """Resize images to size using bilinear interpolation.

    Args:
        images (Tensor): The input images.
        size (tuple[int, int]): The new size (height, width).
        align_corners (bool): Whether to align corners.

    Returns:
        Tensor: The resized images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "ResizeBilinear", images.data, size=size, align_corners=align_corners
        )
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node(
        "ResizeBilinear", [images], {"size": size, "align_corners": align_corners}, (), DType.Int32
    )


def resize_nearest(images: Tensor, size: tuple[int, int], align_corners: bool = False) -> Tensor:
    """Resize images to size using nearest neighbor interpolation.

    Args:
        images (Tensor): The input images.
        size (tuple[int, int]): The new size (height, width).
        align_corners (bool): Whether to align corners.

    Returns:
        Tensor: The resized images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "ResizeNearest", images.data, size=size, align_corners=align_corners
        )
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node(
        "ResizeNearest", [images], {"size": size, "align_corners": align_corners}, (), DType.Int32
    )


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
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node(
        "CropAndResize",
        [images, boxes, box_indices],
        {"crop_size": crop_size, "method": method, "extrapolation_value": extrapolation_value},
        (),
        DType.Int32,
    )


def affine_transform(images: Tensor, transforms: Tensor, interpolation: str = "nearest") -> Tensor:
    """Applies the given 2D affine transforms to the given images.

    Args:
        images (Tensor): Input images.
        transforms (Tensor): Transform matrices.
        interpolation (str): Interpolation method.

    Returns:
        Tensor: Transformed images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "AffineTransform", images.data, transforms.data, interpolation=interpolation
        )
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node(
        "AffineTransform", [images, transforms], {"interpolation": interpolation}, (), DType.Int32
    )


def affine_generator(batch_size: int, angles: Tensor, shears: Tensor, zooms: Tensor) -> Tensor:
    """Constructs 2D/3D affine matrices from angles, shears, and zoom factors.

    Args:
        batch_size (int): The batch size.
        angles (Tensor): Rotation angles.
        shears (Tensor): Shear values.
        zooms (Tensor): Zoom factors.

    Returns:
        Tensor: Generated affine matrices.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "AffineGenerator",
            batch_size=batch_size,
            angles=angles.data,
            shears=shears.data,
            zooms=zooms.data,
        )
        return Tensor(backend.array(data), backend.array(data).shape, angles.dtype, angles.device)
    return _emit_shape_node(
        "AffineGenerator", [angles, shears, zooms], {"batch_size": batch_size}, (), angles.dtype
    )


def flip_left_right(images: Tensor) -> Tensor:
    """Flips images horizontally.

    Args:
        images (Tensor): Input images.

    Returns:
        Tensor: Flipped images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("FlipLeftRight", images.data)
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node("FlipLeftRight", [images], {}, (), DType.Int32)


def flip_up_down(images: Tensor) -> Tensor:
    """Flips images vertically.

    Args:
        images (Tensor): Input images.

    Returns:
        Tensor: Flipped images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("FlipUpDown", images.data)
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node("FlipUpDown", [images], {}, (), DType.Int32)


def perspective_transform(
    images: Tensor,
    start_points: Tensor,
    end_points: Tensor,
    config_obj: typing.Optional[object] = None,
    **kwargs: object,
) -> Tensor:
    """Applies a perspective transformation to the image(s).

    Args:
        images (Tensor): Input images.
        start_points (Tensor): Source points.
        end_points (Tensor): Target points.
        config_obj (PerspectiveConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: Transformed images.
    """
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import PerspectiveConfig

        config_obj = PerspectiveConfig(
            interpolation=kwargs.get("interpolation", "bilinear"),
            fill_value=kwargs.get("fill_value", 0.0),
            data_format=kwargs.get("data_format", None),
        )

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "PerspectiveTransform",
            images.data,
            start_points.data,
            end_points.data,
            config=config_obj,
        )
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node(
        "PerspectiveTransform",
        [images, start_points, end_points],
        {"config": config_obj},
        (),
        DType.Int32,
    )


def elastic_transform(
    images: Tensor,
    displacement: Tensor,
    config_obj: typing.Optional[object] = None,
    **kwargs: object,
) -> Tensor:
    """Applies an elastic transformation to the image(s).

    Args:
        images (Tensor): Input images.
        displacement (Tensor): Displacement field (dy, dx).
        config_obj (ElasticConfig | None): Configuration.
        **kwargs: Backward compatibility arguments.

    Returns:
        Tensor: Transformed images.
    """
    if config_obj is None:
        from ml_switcheroo_compiler.ops.configs import ElasticConfig

        config_obj = ElasticConfig(
            interpolation=kwargs.get("interpolation", "bilinear"),
            fill_value=kwargs.get("fill_value", 0.0),
            data_format=kwargs.get("data_format", None),
        )

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "ElasticTransform",
            images.data,
            displacement.data,
            config=config_obj,
        )
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node(
        "ElasticTransform",
        [images, displacement],
        {"config": config_obj},
        (),
        DType.Int32,
    )


def extract_bounding_boxes(
    images: Tensor,
    boxes: Tensor,
    box_indices: Tensor,
    config_obj: typing.Optional[object] = None,
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
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "ExtractBoundingBoxes",
            images.data,
            boxes.data,
            box_indices.data,
            config=config_obj,
        )
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node(
        "ExtractBoundingBoxes",
        [images, boxes, box_indices],
        {
            "config": config_obj,
        },
        (),
        DType.Int32,
    )


def resize_bicubic(images: Tensor, size: tuple[int, int], align_corners: bool = False) -> Tensor:
    """Resize images to size using bicubic interpolation.

    Args:
        images (Tensor): The input images.
        size (tuple[int, int]): The new size (height, width).
        align_corners (bool): Whether to align corners.

    Returns:
        Tensor: The resized images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "ResizeBicubic", images.data, size=size, align_corners=align_corners
        )
        return Tensor(backend.array(data), backend.array(data).shape, images.dtype, images.device)
    return _emit_shape_node(
        "ResizeBicubic",
        [images],
        {"size": size, "align_corners": align_corners},
        (),
        images.dtype,
    )


def resize_lanczos3(images: Tensor, size: tuple[int, int], align_corners: bool = False) -> Tensor:
    """Resize images to size using lanczos3 interpolation.

    Args:
        images (Tensor): The input images.
        size (tuple[int, int]): The new size (height, width).
        align_corners (bool): Whether to align corners.

    Returns:
        Tensor: The resized images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "ResizeLanczos3", images.data, size=size, align_corners=align_corners
        )
        return Tensor(backend.array(data), backend.array(data).shape, images.dtype, images.device)
    return _emit_shape_node(
        "ResizeLanczos3",
        [images],
        {"size": size, "align_corners": align_corners},
        (),
        images.dtype,
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
        return Tensor(backend.array(data), backend.array(data).shape, images.dtype, images.device)
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
        return Tensor(backend.array(data), backend.array(data).shape, images.dtype, images.device)
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
