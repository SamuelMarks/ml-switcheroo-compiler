"""Shared helper functions for symbolic signal processing IR node emission and padding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue, ShapeDim
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


def _emit_signal_node(
    op_type: str,
    inputs: Sequence[Tensor | OpValue],
    attrs: Mapping[str, OpValue] | None,
    out_shape: Sequence[ShapeDim],
    dtype: DType | str | None,
) -> OpValue:
    """Emit a symbolic signal processing IR node and return its proxy `Tensor`.

    Args:
        op_type (str): Canonical IR operation type name.
        inputs (Sequence[Tensor | OpValue]): Input tensor operands.
        attrs (Mapping[str, OpValue] | None): Optional mapping of operation attributes.
        out_shape (Sequence[ShapeDim]): Inferred output tensor shape.
        dtype (DType | str | None): Output tensor data type.

    Returns:
        OpValue: Symbolic `Tensor` wrapping the emitted `LogicalNode`.
    """
    return _emit_linalg_node(op_type, inputs, attrs, [out_shape], [dtype])


def _calculate_padding(mode: str, boundary: str, fillvalue: float) -> dict[str, str | float]:
    """Build the padding configuration attribute dictionary for 2D signal convolution.

    Args:
        mode (str): Convolution output size mode (`'full'`, `'valid'`, or `'same'`).
        boundary (str): Boundary handling mode (`'fill'`, `'wrap'`, or `'symm'`).
        fillvalue (float): Constant fill value used when `boundary == 'fill'`.

    Returns:
        dict[str, str | float]: Padding configuration attribute dictionary.
    """
    return {"mode": mode, "boundary": boundary, "fillvalue": fillvalue}
