"""Raw Operations mapping strategy."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


class RawOp(OpDef):
    """Define base class for all Raw operations.

    This provides a strategy to support raw TensorFlow operations dynamically
    by falling back to eager execution or mapping them to higher-level IR nodes
    where applicable.
    """

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape dynamically or fallback to unknown.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
        object: Result.
        """
        return ()


@register_op("RawSwitch")
class RawSwitch(RawOp):
    """Dynamic control flow RawSwitch."""

    op_name = "RawSwitch"


@register_op("RawMerge")
class RawMerge(RawOp):
    """Dynamic control flow RawMerge."""

    op_name = "RawMerge"


@register_op("RawConv2D")
class RawConv2D(RawOp):
    """Raw Conv2D mapping."""

    op_name = "RawConv2D"


@register_op("RawMatMul")
class RawMatMul(RawOp):
    """Raw MatMul mapping."""

    op_name = "RawMatMul"
