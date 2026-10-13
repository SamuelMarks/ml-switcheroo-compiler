# ruff: noqa: E402, E501, F403
"""Signal processing operations."""

from __future__ import annotations

from dataclasses import dataclass  # noqa: F401
from typing import Optional  # noqa: F401

from ml_switcheroo_compiler.backends.registry import get_active_backend  # noqa: F401
from ml_switcheroo_compiler.core.config import config  # noqa: F401
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # noqa: F401
from ml_switcheroo_compiler.ops.base import OpDef, register_op  # noqa: F401
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node  # noqa: F401


def _emit_signal_node(
    op_type: str,
    inputs: list[Tensor],
    attrs: dict[str, int | float | str | bool | tuple[int, ...]],
    out_shape: tuple[int, ...],
    dtype: str,
) -> Tensor | tuple[Tensor, ...]:
    """Emit a signal node.

    Args:
        op_type (str): The op_type parameter.
        inputs (list): The inputs parameter.
        attrs (dict): The attrs parameter.
        out_shape (tuple): The out_shape parameter.
        dtype (str): The dtype parameter.

    Returns:
        Tensor | tuple[Tensor, ...]: Result.
    """
    return _emit_linalg_node(op_type, inputs, attrs, [out_shape], [dtype])


def _calculate_padding(mode: str, boundary: str, fillvalue: float) -> dict[str, str | float]:
    """Calculate padding configuration for convolve2d.

    Args:
        mode (str): Padding mode.
        boundary (boundary): Boundary condition.
        fillvalue (float): Fill value for 'fill' boundary.

    Returns:
        dict[str, str | float]: Padding configuration dictionary.
    """
    return {"mode": mode, "boundary": boundary, "fillvalue": fillvalue}
