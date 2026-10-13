# ruff: noqa: E402, E501, F403
"""Numerical anomaly detection and traceback."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor


def format_traceback(exc: Exception) -> str:
    """Format an exception traceback.

    Args:
        exc (Exception): The exception to format.

    Returns:
        str: Formatted traceback string.
    """
    return f"TracebackReconstructor: {exc!s}"


def check_numerical_anomaly(tensor: Tensor | np.ndarray | None) -> None:
    """Check a tensor for numerical anomalies.

    Args:
        tensor (Tensor | np.ndarray | None): The tensor to check.

    Raises:
        ValueError: If NaN or Inf is found.
    """
    if getattr(tensor, "data", None) is None:
        return

    try:
        from ml_switcheroo_compiler import ops

        if bool(ops.any(ops.isnan(tensor))) or bool(ops.any(ops.isinf(tensor))):
            msg = "Tensor contains NaN or Inf."
            raise ValueError(msg)
    except (TypeError, ValueError) as e:
        if "NaN or Inf" in str(e):
            raise
