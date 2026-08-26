# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Signal processing operations."""

from dataclasses import dataclass
from typing import Optional

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def _emit_signal_node(
    op_type: str,
    inputs: list[Tensor],
    attrs,
    out_shape: tuple[int, ...],
    dtype: str,
):
    """Emit a signal node.

    Args:
        op_type (str): The op_type parameter.
        inputs (list): The inputs parameter.
        attrs (dict): The attrs parameter.
        out_shape (tuple): The out_shape parameter.
        dtype (str): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    return _emit_linalg_node(op_type, inputs, attrs, [out_shape], [dtype])


def _calculate_padding(mode: str, boundary: str, fillvalue: float):
    """Calculate padding configuration for convolve2d.

    Args:
        mode (str): Padding mode.
        boundary (str): Boundary condition.
        fillvalue (float): Fill value for 'fill' boundary.

    Returns:
        dict[str, object]: Padding configuration dictionary.
    """
    return {"mode": mode, "boundary": boundary, "fillvalue": fillvalue}
