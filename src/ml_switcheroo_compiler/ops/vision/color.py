"""Vision and Image processing operations."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def rgb_to_hsv(images: Tensor) -> Tensor:
    """Converts one or more images from RGB to HSV.

    Args:
        images (Tensor): Input images.

    Returns:
        Tensor: HSV images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("RgbToHsv", images.data)
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node("RgbToHsv", [images], {}, (), DType.Int32)


def hsv_to_rgb(images: Tensor) -> Tensor:
    """Converts one or more images from HSV to RGB.

    Args:
        images (Tensor): Input images.

    Returns:
        Tensor: RGB images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("HsvToRgb", images.data)
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node("HsvToRgb", [images], {}, (), DType.Int32)


def adjust_hue(images: Tensor, delta: float) -> Tensor:
    """Adjust hue of RGB images.

    Args:
        images (Tensor): Input images.
        delta (float): How much to add to the hue channel.

    Returns:
        Tensor: Adjusted images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("AdjustHue", images.data, delta=delta)
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node("AdjustHue", [images], {"delta": delta}, (), DType.Int32)


def adjust_saturation(images: Tensor, saturation_factor: float) -> Tensor:
    """Adjust saturation of RGB images.

    Args:
        images (Tensor): Input images.
        saturation_factor (float): Multiplier for the saturation channel.

    Returns:
        Tensor: Adjusted images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "AdjustSaturation", images.data, saturation_factor=saturation_factor
        )
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node(
        "AdjustSaturation", [images], {"saturation_factor": saturation_factor}, (), DType.Int32
    )


def adjust_contrast(images: Tensor, contrast_factor: float) -> Tensor:
    """Adjust contrast of RGB or grayscale images.

    Args:
        images (Tensor): Input images.
        contrast_factor (float): Multiplier for the contrast.

    Returns:
        Tensor: Adjusted images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("AdjustContrast", images.data, contrast_factor=contrast_factor)
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node(
        "AdjustContrast", [images], {"contrast_factor": contrast_factor}, (), DType.Int32
    )


def adjust_brightness(images: Tensor, delta: float) -> Tensor:
    """Adjust brightness of images.

    Args:
        images (Tensor): Input images.
        delta (float): Amount to add to pixel values.

    Returns:
        Tensor: Adjusted images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("AdjustBrightness", images.data, delta=delta)
        return Tensor(backend.array(data), backend.array(data).shape, DType.Int32, images.device)
    return _emit_shape_node("AdjustBrightness", [images], {"delta": delta}, (), DType.Int32)
