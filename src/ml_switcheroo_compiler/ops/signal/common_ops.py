"""Shared helper functions for symbolic signal processing IR node emission and padding."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, as_tensor_obj
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


def _emit_signal_node(
    op_type: str,
    inputs: Sequence[Tensor | TensorOrScalar],
    attrs: dict[str, TensorOrScalar] | None,
    out_shape: Sequence[int | str],
    dtype: DType | str | None,
) -> Tensor:
    """Emit a symbolic signal processing IR node and return its proxy `Tensor`.

    Args:
        op_type (str): Canonical IR operation type name.
        inputs (Sequence[Tensor | TensorOrScalar]): Input tensor operands.
        attrs (dict[str, TensorOrScalar] | None): Optional dictionary of operation attributes.
        out_shape (Sequence[int | str]): Inferred output tensor shape.
        dtype (DType | str | None): Output tensor data type.

    Returns:
        Tensor: Symbolic `Tensor` wrapping the emitted `LogicalNode`.
    """
    return as_tensor_obj(_emit_linalg_node(op_type, inputs, attrs, [out_shape], [dtype]))


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
