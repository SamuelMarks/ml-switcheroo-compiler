"""Numerical anomaly detection and traceback."""


class TracebackReconstructor:
    """Utility to reconstruct tracebacks."""

    @staticmethod
    def format_traceback(exc: Exception) -> str:
        """Format an exception traceback.

        Args:
            exc (Exception): The exception.

        Returns:
            str: Formatted traceback string.
        """
        return f"TracebackReconstructor: {str(exc)}"


class NumericalAnomalyDetector:
    """Detector for NaN and Inf cascades."""

    @staticmethod
    def check(tensor: object) -> None:
        """Check a tensor for numerical anomalies.

        Args:
            tensor (Any): The tensor to check.

        Raises:
            ValueError: If NaN or Inf is found.
        """
        if tensor.data is None:
            return

        import numpy as np

        try:
            arr = np.asarray(tensor.data)
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ValueError("Tensor contains NaN or Inf.")
        except (TypeError, ValueError) as e:
            if "NaN or Inf" in str(e):
                raise
            pass
