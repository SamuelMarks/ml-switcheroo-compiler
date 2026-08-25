# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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
