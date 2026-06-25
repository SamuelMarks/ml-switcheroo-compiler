"""Vision operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
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
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("RgbToHsv")()(images, dtype=DType.Int32)


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
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("HsvToRgb")()(images, dtype=DType.Int32)


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
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
    from ml_switcheroo_compiler.ops.base import get_op

    kwargs = {"delta": delta}
    return get_op("AdjustHue")()(images, dtype=DType.Int32, **kwargs)


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
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
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
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
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
        return Tensor(
            backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device)
        )
    from ml_switcheroo_compiler.ops.base import get_op

    kwargs = {"delta": delta}
    return get_op("AdjustBrightness")()(images, dtype=DType.Int32, **kwargs)


def rgb_to_grayscale(images: Tensor, data_format: str = "channels_last") -> Tensor:
    """Convert RGB images to grayscale.

    Args:
        images (Tensor): Input image or batch of images. Must be 3D or 4D.
        data_format (str): A string specifying the data format of the input tensor.

    Returns:
        Tensor: Grayscale image or batch of grayscale images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("RgbToGrayscale", images.data, data_format=data_format)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RgbToGrayscale", [images], {"data_format": data_format}, (), images.dtype
    )


def random_color_jitter(
    images: Tensor,
    **kwargs: object,
) -> Tensor:
    """Randomly applies color jitter to images.

    Args:
        images (Tensor): Input images.
        **kwargs (object): Additional keyword arguments.

    Returns:
        Tensor: Jittered images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "RandomColorJitter",
            images.data,
            **kwargs,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node(
        "RandomColorJitter",
        [images],
        {
            **kwargs,
        },
        (),
        images.dtype,
    )


def solarize(images: Tensor, threshold: float = 0.5, value_range: tuple = (0, 255)) -> Tensor:
    """Solarize images (invert all pixel values above a threshold).

    Args:
        images (Tensor): Input images.
        threshold (float): Threshold for solarization.
        value_range (tuple, optional): The range of pixel values. Defaults to (0, 255).

    Returns:
        Tensor: Solarized images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Solarize", images.data, threshold=threshold)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op

    kwargs = {"threshold": threshold, "value_range": value_range}
    return get_op("Solarize")()(images, **kwargs)


def invert(images: Tensor, value_range: tuple = (0, 255)) -> Tensor:
    """Invert image pixels.

    Args:
        images (Tensor): Input images.
        value_range (tuple, optional): The range of pixel values. Defaults to (0, 255).

    Returns:
        Tensor: Inverted images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Invert", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("Invert")()(images, value_range=value_range)


def posterize(images: Tensor, bits: int) -> Tensor:
    """Posterize images (reduce the number of bits for each color channel).

    Args:
        images (Tensor): Input images.
        bits (int): Number of bits to keep for each channel.

    Returns:
        Tensor: Posterized images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Posterize", images.data, bits=bits)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op

    kwargs = {"bits": bits}
    return get_op("Posterize")()(images, **kwargs)


def degeneration(images: Tensor, factor: float = 0.0) -> Tensor:
    """Apply degeneration/noise to images.

    Args:
        images (Tensor): Input images.
        factor (float): Factor of degeneration.

    Returns:
        Tensor: Degenerated images.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Degeneration", images.data, factor=factor)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op

    kwargs = {"factor": factor}
    return get_op("Degeneration")()(images, **kwargs)


def augmix(images: Tensor, factor: float = 0.3) -> Tensor:
    """AugMix operation.

    Args:
        images: Input images.
        factor: Factor.

    Returns:
        Tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("AugMix", images.data, factor=factor)  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    kwargs = {"factor": factor}  # pragma: no cover
    return get_op("AugMix")()(images, **kwargs)  # pragma: no cover


def auto_contrast(images: Tensor, value_range: tuple = (0, 255)) -> Tensor:
    """AutoContrast operation.

    Args:
        images: Input images.
        value_range (tuple, optional): The range of pixel values. Defaults to (0, 255).

    Returns:
        Tensor.
    """
    if config.eager_mode:  # pragma: no branch
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("AutoContrast", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    return get_op("AutoContrast")()(images, value_range=value_range)  # pragma: no cover


def rand_augment(images: Tensor, factor: float = 0.5) -> Tensor:
    """RandAugment operation.

    Args:
        images: Input images.
        factor: Factor.

    Returns:
        Tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("RandAugment", images.data, factor=factor)  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    kwargs = {"factor": factor}  # pragma: no cover
    return get_op("RandAugment")()(images, **kwargs)  # pragma: no cover


def random_erasing(images: Tensor, factor: float = 1.0) -> Tensor:
    """RandomErasing operation.

    Args:
        images: Input images.
        factor: Factor.

    Returns:
        Tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("RandomErasing", images.data, factor=factor)  # pragma: no cover
        return Tensor(  # pragma: no cover
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    kwargs = {"factor": factor}  # pragma: no cover
    return get_op("RandomErasing")()(images, **kwargs)  # pragma: no cover


def equalization(images: Tensor) -> Tensor:
    """Equalization operation.

    Args:
        images: Input images.

    Returns:
        Tensor.
    """
    if config.eager_mode:  # pragma: no branch
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Equalization", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    return get_op("Equalization")()(images)  # pragma: no cover


def rgb_to_yiq(images: Tensor) -> Tensor:
    """Converts one or more images from RGB to YIQ.

    Args:
        images (Tensor): Input images.

    Returns:
    Tensor: Images in YIQ space.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("RgbToYiq", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node("RgbToYiq", [images], {}, (), images.dtype)


def yiq_to_rgb(images: Tensor) -> Tensor:
    """Converts one or more images from YIQ to RGB.

    Args:
        images (Tensor): Input images.

    Returns:
    Tensor: Images in RGB space.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("YiqToRgb", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node("YiqToRgb", [images], {}, (), images.dtype)


def rgb_to_yuv(images: Tensor) -> Tensor:
    """Converts one or more images from RGB to YUV.

    Args:
        images (Tensor): Input images.

    Returns:
    Tensor: Images in YUV space.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("RgbToYuv", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node("RgbToYuv", [images], {}, (), images.dtype)


def yuv_to_rgb(images: Tensor) -> Tensor:
    """Converts one or more images from YUV to RGB.

    Args:
        images (Tensor): Input images.

    Returns:
    Tensor: Images in RGB space.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("YuvToRgb", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node("YuvToRgb", [images], {}, (), images.dtype)
