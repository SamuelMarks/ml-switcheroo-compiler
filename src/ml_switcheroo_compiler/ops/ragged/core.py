"""Ragged ops core."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("RaggedDot")
class RaggedDot(OpDef):
    """RaggedDot op."""

    op_name = "RaggedDot"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
        object: Result.
        """
        # Typically returns the shape of the dense representation or similar.
        # For simplicity we
        # return () or a basic shape derived from inputs.

        return ()
