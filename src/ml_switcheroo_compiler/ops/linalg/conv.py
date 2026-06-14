"""Linear algebra operations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sequence

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node

if TYPE_CHECKING:
    from collections.abc import Sequence


def conv_general_dilated(
    lhs: Tensor,
    rhs: Tensor,
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]] | str,
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: object = None,
) -> Tensor:
    """General N-dimensional convolution with support for strides, padding, and dilations.

    Args:
        lhs (Tensor): Left-hand side tensor (input).
        rhs (Tensor): Right-hand side tensor (filters/weights).
        window_strides (Sequence[int]): Strides of the window.
        padding (Sequence[tuple[int, int]] | str): Padding to apply.
        lhs_dilation (Sequence[int] | None): Dilation of the input.
        rhs_dilation (Sequence[int] | None): Dilation of the weights.
        dimension_numbers (object): Dimension numbers specification.

    Returns:
    Tensor: The result of the convolution.

    Raises:
    UnimplementedMathError: If called in eager mode.
    """
    if config.eager_mode:
        msg = "No direct numpy for conv_general_dilated"
        raise UnimplementedMathError(msg)

    inputs = [lhs, rhs]
    attributes = {
        "window_strides": window_strides,
        "padding": padding,
        "lhs_dilation": lhs_dilation,
        "rhs_dilation": rhs_dilation,
        "dimension_numbers": dimension_numbers,
    }

    from ml_switcheroo_compiler.ops.linalg.basic import ConvGeneralDilated

    op = ConvGeneralDilated()
    from ml_switcheroo_compiler.ops.linalg.basic import ConvGeneralDilatedConfig

    cfg = ConvGeneralDilatedConfig(
        window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers
    )
    out_shape = op.infer_shape(lhs, rhs, cfg)

    return _emit_linalg_node("ConvGeneralDilated", inputs, attributes, [out_shape], [lhs.dtype])
