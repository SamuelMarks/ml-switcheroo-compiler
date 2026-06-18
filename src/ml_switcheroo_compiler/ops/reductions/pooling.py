"""Pooling reduction operations."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("CTCLoss")
class CTCLoss(OpDef):
    """Connectionist Temporal Classification Loss."""

    def infer_shape(
        self,
        log_probs: object,
        targets: object,
        input_lengths: object,
        target_lengths: object,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        if hasattr(log_probs, "shape"):
            return (log_probs.shape[1],) if len(log_probs.shape) >= 2 else ()
        return ()


@register_op("FractionalMaxPool2D")
class FractionalMaxPool2D(OpDef):
    """Fractional max pooling 2D."""

    def infer_shape(self, operand: object, output_size: object, **kwargs: object) -> object:
        """Infer shape."""
        if hasattr(operand, "shape") and hasattr(output_size, "__getitem__"):
            s = list(operand.shape)
            s[-2], s[-1] = output_size[0], output_size[1]
            return tuple(s)
        return ()


@register_op("AdaptiveAvgPool2D")
class AdaptiveAvgPool2D(OpDef):
    """Adaptive average pooling 2D."""

    def infer_shape(self, operand: object, output_size: object, **kwargs: object) -> object:
        """Infer shape."""
        if hasattr(operand, "shape") and hasattr(output_size, "__getitem__"):
            s = list(operand.shape)
            s[-2], s[-1] = output_size[0], output_size[1]
            return tuple(s)
        return ()


@register_op("AdaptiveMaxPool2D")
class AdaptiveMaxPool2D(OpDef):
    """Adaptive max pooling 2D."""

    def infer_shape(self, operand: object, output_size: object, **kwargs: object) -> object:
        """Infer shape."""
        if hasattr(operand, "shape") and hasattr(output_size, "__getitem__"):
            s = list(operand.shape)
            s[-2], s[-1] = output_size[0], output_size[1]
            return tuple(s)
        return ()


@register_op("Unfold")
class Unfold(OpDef):
    """Unfold (Im2Col) operator."""

    def infer_shape(self, operand: object, kernel_size: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("Fold")
class Fold(OpDef):
    """Fold (Col2Im) operator."""

    def infer_shape(
        self,
        operand: object,
        output_size: object,
        kernel_size: object,
        **kwargs: object,
    ) -> object:
        """Infer shape."""
        return ()
