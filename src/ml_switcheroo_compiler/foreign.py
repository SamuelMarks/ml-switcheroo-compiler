"""Foreign module integration."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("ForeignCall")
class ForeignCall(OpDef):
    """Universal ForeignCall op for external graphs/modules."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()
