"""Numerical anomaly detection and traceback."""


def format_traceback(exc: Exception) -> str:
    """Format an exception traceback.

    Args:
        exc (Exception): The exception

    Returns:
        str: Formatted traceback string
    """
    return f"TracebackReconstructor: {exc!s}"


def check_numerical_anomaly(tensor: object) -> None:
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
