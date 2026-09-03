"""Module color.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Vision operations."""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, get_op, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def rgb_to_hsv(images: Tensor):
    """Convert one or more images from RGB to HSV.

    Args:
        images (Tensor): Input images.

    Returns:
        Tensor: HSV images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RgbToHsv", images.data)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))

    return get_op("RgbToHsv")()(images, dtype=DType.Int32)


def hsv_to_rgb(images: Tensor):
    """Convert one or more images from HSV to RGB.

    Args:
        images (Tensor): Input images.

    Returns:
        Tensor: RGB images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("HsvToRgb", images.data)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))

    return get_op("HsvToRgb")()(images, dtype=DType.Int32)


def adjust_hue(images: Tensor, delta: float):
    """Adjust hue of RGB images.

    Args:
        images (Tensor): Input images.
        delta (float): How much to add to the hue channel.

    Returns:
        Tensor: Adjusted images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AdjustHue", images.data, delta=delta)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))

    kwargs = {"delta": delta}
    return get_op("AdjustHue")()(images, dtype=DType.Int32, **kwargs)


def adjust_saturation(images: Tensor, saturation_factor: float):
    """Adjust saturation of RGB images.

    Args:
        images (Tensor): Input images.
        saturation_factor (float): Multiplier for the saturation channel.

    Returns:
        Tensor: Adjusted images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AdjustSaturation", images.data, saturation_factor=saturation_factor)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
    return _emit_shape_node("AdjustSaturation", [images], {"saturation_factor": saturation_factor}, (), DType.Int32)


def adjust_contrast(images: Tensor, contrast_factor: float):
    """Adjust contrast of RGB or grayscale images.

    Args:
        images (Tensor): Input images.
        contrast_factor (float): Multiplier for the contrast.

    Returns:
        Tensor: Adjusted images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AdjustContrast", images.data, contrast_factor=contrast_factor)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))
    return _emit_shape_node("AdjustContrast", [images], {"contrast_factor": contrast_factor}, (), DType.Int32)


def adjust_brightness(images: Tensor, delta: float):
    """Adjust brightness of images.

    Args:
        images (Tensor): Input images.
        delta (float): Amount to add to pixel values.

    Returns:
        Tensor: Adjusted images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AdjustBrightness", images.data, delta=delta)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, DType.Int32, images.device))

    kwargs = {"delta": delta}
    return get_op("AdjustBrightness")()(images, dtype=DType.Int32, **kwargs)


def rgb_to_grayscale(images: Tensor, data_format: str = "channels_last"):
    """Convert RGB images to grayscale.

    Args:
        images (Tensor): Input image or batch of images. Must be 3D or 4D.
        data_format (str): A string specifying the data format of the input tensor.

    Returns:
        Tensor: Grayscale image or batch of grayscale images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RgbToGrayscale", images.data, data_format=data_format)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node("RgbToGrayscale", [images], {"data_format": data_format}, (), images.dtype)


def random_color_jitter(
    images: Tensor,
    **kwargs,
):
    """Randomly applies color jitter to images.

    Args:
        images (Tensor): Input images.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        Tensor: Jittered images.
    """
    if config.eager_mode:
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


def solarize(images: Tensor, threshold: float = 0.5, value_range=(0, 255)):
    """Solarize images (invert all pixel values above a threshold).

    Args:
        images (Tensor): The images parameter.
        threshold (float): The threshold parameter.
        value_range (tuple): The value_range parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Solarize", images.data, threshold=threshold)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    kwargs = {"threshold": threshold, "value_range": value_range}
    return get_op("Solarize")()(images, **kwargs)


def invert(images: Tensor, value_range=(0, 255)):
    """Invert image pixels.

    Args:
        images (Tensor): The images parameter.
        value_range (tuple): The value_range parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Invert", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    return get_op("Invert")()(images, value_range=value_range)


def posterize(images: Tensor, bits: int):
    """Posterize images (reduce the number of bits for each color channel).

    Args:
        images (Tensor): Input images.
        bits (int): Number of bits to keep for each channel.

    Returns:
        Tensor: Posterized images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Posterize", images.data, bits=bits)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    kwargs = {"bits": bits}
    return get_op("Posterize")()(images, **kwargs)


def degeneration(images: Tensor, factor: float = 0.0):
    """Apply degeneration/noise to images.

    Args:
        images (Tensor): Input images.
        factor (float): Factor of degeneration.

    Returns:
        Tensor: Degenerated images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Degeneration", images.data, factor=factor)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    kwargs = {"factor": factor}
    return get_op("Degeneration")()(images, **kwargs)


def augmix(images: Tensor, factor: float = 0.3):
    """AugMix operation.

    Args:
        images: Input images.
        factor: Factor.

    Returns:
        Tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AugMix", images.data, factor=factor)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    kwargs = {"factor": factor}
    return get_op("AugMix")()(images, **kwargs)


def auto_contrast(images: Tensor, value_range=(0, 255)):
    """AutoContrast operation.

    Args:
        images (Tensor): The images parameter.
        value_range (tuple): The value_range parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AutoContrast", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    return get_op("AutoContrast")()(images, value_range=value_range)


def rand_augment(images: Tensor, factor: float = 0.5):
    """RandAugment operation.

    Args:
        images: Input images.
        factor: Factor.

    Returns:
        Tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RandAugment", images.data, factor=factor)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    kwargs = {"factor": factor}
    return get_op("RandAugment")()(images, **kwargs)


def random_erasing(images: Tensor, factor: float = 1.0):
    """RandomErasing operation.

    Args:
        images: Input images.
        factor: Factor.

    Returns:
        Tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RandomErasing", images.data, factor=factor)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    kwargs = {"factor": factor}
    return get_op("RandomErasing")()(images, **kwargs)


def equalization(images: Tensor):
    """Equalization operation.

    Args:
        images: Input images.

    Returns:
        Tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Equalization", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )

    return get_op("Equalization")()(images)


def rgb_to_yiq(images: Tensor):
    """Convert one or more images from RGB to YIQ.

    Args:
        images (Tensor): The images parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RgbToYiq", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node("RgbToYiq", [images], {}, (), images.dtype)


def yiq_to_rgb(images: Tensor):
    """Convert one or more images from YIQ to RGB.

    Args:
        images (Tensor): The images parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("YiqToRgb", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node("YiqToRgb", [images], {}, (), images.dtype)


def rgb_to_yuv(images: Tensor):
    """Convert one or more images from RGB to YUV.

    Args:
        images (Tensor): The images parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RgbToYuv", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node("RgbToYuv", [images], {}, (), images.dtype)


def yuv_to_rgb(images: Tensor):
    """Convert one or more images from YUV to RGB.

    Args:
        images (Tensor): The images parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("YuvToRgb", images.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images.dtype, images.device),
        )
    return _emit_shape_node("YuvToRgb", [images], {}, (), images.dtype)


@register_op("AdjustBrightness")
class AdjustBrightness(OpDef):
    """AdjustBrightness operation."""

    op_name = "AdjustBrightness"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
        images (Any): The images parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())


@register_op("AdjustContrast")
class AdjustContrast(OpDef):
    """AdjustContrast operation."""

    op_name = "AdjustContrast"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())


@register_op("AdjustHue")
class AdjustHue(OpDef):
    """AdjustHue operation."""

    op_name = "AdjustHue"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())


@register_op("AdjustSaturation")
class AdjustSaturation(OpDef):
    """AdjustSaturation operation."""

    op_name = "AdjustSaturation"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())


@register_op("AugMix")
class AugMix(OpDef):
    """AugMix operation."""

    op_name = "AugMix"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())


@register_op("AutoContrast")
class AutoContrast(OpDef):
    """AutoContrast operation."""

    op_name = "AutoContrast"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())


@register_op("Equalization")
class Equalization(OpDef):
    """Equalization operation."""

    op_name = "Equalization"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())


@register_op("Invert")
class Invert(OpDef):
    """Invert operation."""

    op_name = "Invert"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())


@register_op("Posterize")
class Posterize(OpDef):
    """Posterize operation."""

    op_name = "Posterize"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())


@register_op("RgbToGrayscale")
class RgbToGrayscale(OpDef):
    """RgbToGrayscale operation."""

    op_name = "RgbToGrayscale"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        s = list(getattr(images, "shape", ()))
        if len(s) > 0:
            data_format = kwargs.get("data_format", "channels_last")
            if data_format == "channels_first":
                s[-3] = 1
            else:
                s[-1] = 1
        return tuple(s)


@register_op("Solarize")
class Solarize(OpDef):
    """Solarize operation."""

    op_name = "Solarize"

    def infer_shape(self, images, *args, **kwargs):
        """Infer shape.

        Args:
            images (Any): The images parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return getattr(images, "shape", ())
