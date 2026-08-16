# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Ragged ops core."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("RaggedDot")
class RaggedDot(OpDef):
    """RaggedDot op."""

    op_name = "RaggedDot"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        # Typically returns the shape of the dense representation or similar.
        # For simplicity we
        # return () or a basic shape derived from inputs.

        return ()
