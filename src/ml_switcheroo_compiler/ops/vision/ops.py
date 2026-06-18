"""Vision operations class definitions."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


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
