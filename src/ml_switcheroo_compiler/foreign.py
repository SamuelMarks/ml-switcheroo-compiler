"""Foreign module integration."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("ForeignCall")
class ForeignCall(OpDef):
    """Universal ForeignCall op for external graphs/modules."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer the shape of the output for ForeignCall.

        Args:
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            tuple[int, ...]: The inferred shape.
        """
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()
