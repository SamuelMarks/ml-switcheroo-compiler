"""Foreign module integration."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op  # pragma: no cover


@register_op("ForeignCall")  # pragma: no cover
class ForeignCall(OpDef):  # pragma: no cover
    """Universal ForeignCall op for external graphs/modules."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:  # pragma: no cover
        """Infer shape."""
        if args and hasattr(args[0], "shape"):  # pragma: no cover
            return args[0].shape  # pragma: no cover
        return ()  # pragma: no cover
