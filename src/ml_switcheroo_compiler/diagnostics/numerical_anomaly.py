"""Numerical anomaly detection and traceback."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.shape import TensorOrScalar

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


@typing.runtime_checkable
class _TensorDataProtocol(typing.Protocol):
    """Protocol for tensor objects exposing a `.data` attribute."""

    data: TensorOrScalar


def format_traceback(exc: Exception) -> str:
    """Format an exception traceback.

    Args:
        exc (Exception): The exception to format.

    Returns:
        str: Formatted traceback string.
    """
    return f"TracebackReconstructor: {exc!s}"


def check_numerical_anomaly(tensor: Tensor | None) -> None:
    """Check a tensor for numerical anomalies.

    Args:
        tensor (Tensor | None): The tensor to check.

    Raises:
        ValueError: If NaN or Inf is found.
    """
    if tensor is None or not isinstance(tensor, _TensorDataProtocol) or tensor.data is None:
        return

    try:
        from ml_switcheroo_compiler import ops

        if bool(ops.any(ops.isnan(tensor))) or bool(ops.any(ops.isinf(tensor))):
            msg = "Tensor contains NaN or Inf."
            raise ValueError(msg)
    except (TypeError, ValueError) as e:
        if "NaN or Inf" in str(e):
            raise
