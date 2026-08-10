from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Vision operations."""


from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def resize_bilinear(images: Tensor, size: tuple[int, int], align_corners: bool = False) -> Any:
    """Resize images to size using bilinear interpolation.

    Args:
        images (Tensor): The input images.
        size (tuple[int, int]): The new size (height, width).
        align_corners (bool): Whether to align corners.

    Returns:
        Tensor: The resized images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("ResizeBilinear", images.data, size=size, align_corners=align_corners)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
    return _emit_shape_node("ResizeBilinear", [images], {"size": size, "align_corners": align_corners}, (), DType.Int32)


def resize_nearest(images: Tensor, size: tuple[int, int], align_corners: bool = False) -> Any:
    """Resize images to size using nearest neighbor interpolation.

    Args:
        images (Tensor): The input images.
        size (tuple[int, int]): The new size (height, width).
        align_corners (bool): Whether to align corners.

    Returns:
        Tensor: The resized images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("ResizeNearest", images.data, size=size, align_corners=align_corners)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
    return _emit_shape_node("ResizeNearest", [images], {"size": size, "align_corners": align_corners}, (), DType.Int32)


def resize_bicubic(images: Tensor, size: tuple[int, int], align_corners: bool = False) -> Any:
    """Resize images to size using bicubic interpolation.

    Args:
        images (Tensor): The input images.
        size (tuple[int, int]): The new size (height, width).
        align_corners (bool): Whether to align corners.

    Returns:
        Tensor: The resized images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("ResizeBicubic", images.data, size=size, align_corners=align_corners)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "ResizeBicubic",
        [images],
        {"size": size, "align_corners": align_corners},
        (),
        images.dtype,
    )


def resize_lanczos3(images: Tensor, size: tuple[int, int], align_corners: bool = False) -> Any:
    """Resize images to size using lanczos3 interpolation.

    Args:
        images (Tensor): The input images.
        size (tuple[int, int]): The new size (height, width).
        align_corners (bool): Whether to align corners.

    Returns:
        Tensor: The resized images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("ResizeLanczos3", images.data, size=size, align_corners=align_corners)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "ResizeLanczos3",
        [images],
        {"size": size, "align_corners": align_corners},
        (),
        images.dtype,
    )


def resize(images: Tensor, size: tuple[int, int], method: str = "bilinear", antialias: bool = False) -> Any:
    """Resize images to size using the specified method.

    Args:
        images (Tensor): The input images.
        size (tuple[int, int]): The new size (height, width).
        method (str): Interpolation method (bilinear, nearest, bicubic, lanczos3, lanczos5).
        antialias (bool): Whether to apply antialiasing.

    Returns:
        Tensor: The resized images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Resize", images.data, size=size, method=method, antialias=antialias)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "Resize",
        [images],
        {"size": size, "method": method, "antialias": antialias},
        (),
        images.dtype,
    )


def map_coordinates(
    input: Tensor,
    coordinates: Tensor,
    order: int,
    fill_mode: str = "half_pixel",
    cval: float = 0.0,
) -> Any:
    """Map coordinates.

    Args:
        input: Input tensor.
        coordinates: Coordinates tensor.
        order: Interpolation order.
        fill_mode: Fill mode.
        cval: Constant value for fill.

    Returns:
        Tensor: Interpolated tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "MapCoordinates",
            input.data,
            coordinates.data,
            order=order,
            fill_mode=fill_mode,
            cval=cval,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, input.dtype, input.device),
        )
    return _emit_shape_node(
        "MapCoordinates",
        [input, coordinates],
        {
            "order": order,
            "fill_mode": fill_mode,
            "cval": cval,
        },
        (),
        input.dtype,
    )
