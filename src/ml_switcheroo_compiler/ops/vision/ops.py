"""Vision ops."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_3
from ml_switcheroo_compiler.core.constants import MAGIC_VAL_4

from ml_switcheroo_compiler.ops.base import OpDef, register_op


"""Vision operations class definitions."""


@register_op("ResizeBilinear")
class ResizeBilinear(OpDef):
    """ResizeBilinear op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("ResizeNearest")
class ResizeNearest(OpDef):
    """ResizeNearest op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("CropAndResize")
class CropAndResize(OpDef):
    """CropAndResize op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("RgbToHsv")
class RgbToHsv(OpDef):
    """RgbToHsv op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("HsvToRgb")
class HsvToRgb(OpDef):
    """HsvToRgb op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("AdjustHue")
class AdjustHue(OpDef):
    """AdjustHue op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("AdjustSaturation")
class AdjustSaturation(OpDef):
    """AdjustSaturation op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("AdjustContrast")
class AdjustContrast(OpDef):
    """AdjustContrast op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("AffineTransform")
class AffineTransform(OpDef):
    """AffineTransform op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("AffineGenerator")
class AffineGenerator(OpDef):
    """AffineGenerator op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("FlipLeftRight")
class FlipLeftRight(OpDef):
    """FlipLeftRight op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("FlipUpDown")
class FlipUpDown(OpDef):
    """FlipUpDown op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("AdjustBrightness")
class AdjustBrightness(OpDef):
    """AdjustBrightness op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("PerspectiveTransform")
class PerspectiveTransform(OpDef):
    """PerspectiveTransform op."""

    def infer_shape(
        self, images: object, start_points: object, end_points: object, **kwargs: object
    ) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("ElasticTransform")
class ElasticTransform(OpDef):
    """ElasticTransform op."""

    def infer_shape(
        self, images: object, displacement: object, **kwargs: object
    ) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("GaussianBlur")
class GaussianBlur(OpDef):
    """GaussianBlur op."""

    def infer_shape(self, images: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("MedianFilter")
class MedianFilter(OpDef):
    """MedianFilter op."""

    def infer_shape(self, images: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("ExtractBoundingBoxes")
class ExtractBoundingBoxes(OpDef):
    """ExtractBoundingBoxes op."""

    def infer_shape(
        self, images: object, boxes: object, box_indices: object, **kwargs: object
    ) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("IoU")
class IoU(OpDef):
    """Intersection-Over-Union op."""

    def infer_shape(self, boxes1: object, boxes2: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("NonMaxSuppression")
class NonMaxSuppression(OpDef):
    """Non-Max Suppression op."""

    def infer_shape(
        self, boxes: object, scores: object, max_output_size: object, **kwargs: object
    ) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("ResizeBicubic")
class ResizeBicubic(OpDef):
    """ResizeBicubic op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("ResizeLanczos5")
class ResizeLanczos5(OpDef):
    """ResizeLanczos5 op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("ResizeLanczos3")
class ResizeLanczos3(OpDef):
    """ResizeLanczos3 op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("Crop")
class Crop(OpDef):
    """Crop op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("PadToBoundingBox")
class PadToBoundingBox(OpDef):
    """PadToBoundingBox op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("RandomFlip")
class RandomFlipOp(OpDef):
    """RandomFlip operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("RandomRotation")
class RandomRotationOp(OpDef):
    """RandomRotation operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("RandomCrop")
class RandomCropOp(OpDef):
    """RandomCrop operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        size = kwargs.get("size", (0, 0))
        shape = list(args[0])
        if len(shape) == MAGIC_VAL_3:  # pragma: no branch
            shape[0] = size[0]  # pragma: no cover
            shape[1] = size[1]  # pragma: no cover
        elif len(shape) == MAGIC_VAL_4:
            shape[1] = size[0]
            shape[2] = size[1]
        return tuple(shape)


@register_op("RgbToGrayscale")
class RgbToGrayscaleOp(OpDef):
    """RgbToGrayscale operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        shape = list(args[0])
        data_format = kwargs.get("data_format", "channels_last")
        if data_format == "channels_last":  # pragma: no branch
            shape[-1] = 1
        else:
            shape[-3] = 1  # pragma: no cover
        return tuple(shape)


@register_op("RandomColorJitter")
class RandomColorJitter(OpDef):
    """RandomColorJitter op."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("Solarize")
class Solarize(OpDef):
    """Solarize op."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("Invert")
class Invert(OpDef):
    """Invert op."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("Posterize")
class Posterize(OpDef):
    """Posterize op."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("Degeneration")
class Degeneration(OpDef):
    """Degeneration op."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("Sharpen")
class Sharpen(OpDef):
    """Sharpen op."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("Mixup")
class Mixup(OpDef):
    """Mixup op."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("Cutmix")
class Cutmix(OpDef):
    """Cutmix op."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("AugMix")
class AugMix(OpDef):
    """AugMix operation."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0]


@register_op("AutoContrast")
class AutoContrast(OpDef):
    """AutoContrast operation."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0]


@register_op("RandAugment")
class RandAugment(OpDef):
    """RandAugment operation."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0]


@register_op("RandomErasing")
class RandomErasing(OpDef):
    """RandomErasing operation."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0]


@register_op("Equalization")
class Equalization(OpDef):
    """Equalization operation."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0]


@register_op("RandomZoom")
class RandomZoomOp(OpDef):
    """RandomZoom operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("RandomShear")
class RandomShearOp(OpDef):
    """RandomShear operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("RandomTranslation")
class RandomTranslationOp(OpDef):
    """RandomTranslation operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("RandomPerspective")
class RandomPerspectiveOp(OpDef):
    """RandomPerspective operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("RandomElasticTransform")
class RandomElasticTransformOp(OpDef):
    """RandomElasticTransform operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("RandomGaussianBlur")
class RandomGaussianBlurOp(OpDef):
    """RandomGaussianBlur operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]


@register_op("RandomSharpness")
class RandomSharpnessOp(OpDef):
    """RandomSharpness operation definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0]
