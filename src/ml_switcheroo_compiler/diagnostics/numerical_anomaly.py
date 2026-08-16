# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numerical anomaly detection and traceback."""

from typing import Any


def format_traceback(exc: Exception) -> str:
    """Format an exception traceback.

    Args:
        exc (Exception): The exception

    Returns:
        str: Formatted traceback string
    """
    return f"TracebackReconstructor: {exc!s}"


def check_numerical_anomaly(tensor: Any) -> None:
    """Check a tensor for numerical anomalies.

    Args:
        tensor (object): The tensor to check

    Raises:
        ValueError: If NaN or Inf is found
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
