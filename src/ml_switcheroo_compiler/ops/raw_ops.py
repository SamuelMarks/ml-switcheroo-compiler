# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Raw Operations mapping strategy."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op


class RawOp(OpDef):
    """Define base class for all Raw operations.

    This provides a strategy to support raw TensorFlow operations dynamically
    by falling back to eager execution or mapping them to higher-level IR nodes
    where applicable.
    """

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape dynamically or fallback to unknown.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
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
