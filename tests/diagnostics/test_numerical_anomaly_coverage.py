"""Tests for test_numerical_anomaly_coverage."""

from __future__ import annotations

import numpy as np
import pytest

import ml_switcheroo_compiler.diagnostics.numerical_anomaly as num_mod
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


class _DummyBadShape:
    """Shape object whose iterator raises TypeError."""

    def __iter__(self) -> _DummyBadShape:
        """Return self as iterator.

        Returns:
            _DummyBadShape: Self.
        """
        return self

    def __next__(self) -> int:
        """Raise TypeError on iteration.

        Raises:
            TypeError: Simulated invalid dimension.
        """
        raise TypeError("Not iterable dimension")


def test_numerical_anomaly_exhaustive() -> None:
    """Verify traceback formatting and anomaly detection with NaN, Inf, and clean arrays."""
    from ml_switcheroo_compiler.core.config import config

    # 1. format_traceback
    err = ValueError("Test error message")
    tb_str = num_mod.format_traceback(err)
    assert tb_str == "TracebackReconstructor: Test error message"

    # 2. None tensor or tensor with no data
    num_mod.check_numerical_anomaly(None)

    # 3. Clean tensor in eager mode
    prev_eager = config.eager_mode
    try:
        config.eager_mode = True

        clean_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        clean_t = Tensor(clean_arr, TensorConfig((3,), "float32", "cpu"))
        num_mod.check_numerical_anomaly(clean_t)

        # 4. Tensor containing NaN
        nan_arr = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        nan_t = Tensor(nan_arr, TensorConfig((3,), "float32", "cpu"))
        with pytest.raises(ValueError, match="Tensor contains NaN or Inf."):
            num_mod.check_numerical_anomaly(nan_t)

        # 5. Tensor containing Inf
        inf_arr = np.array([1.0, float("inf"), 3.0], dtype=np.float32)
        inf_t = Tensor(inf_arr, TensorConfig((3,), "float32", "cpu"))
        with pytest.raises(ValueError, match="Tensor contains NaN or Inf."):
            num_mod.check_numerical_anomaly(inf_t)

        # 6. Tensor with object data raising non-anomaly error
        class _NonArray:
            """Non-array dummy class for testing exception handling."""

        bad_t = Tensor(_NonArray(), TensorConfig((1,), "float32", "cpu"))
        num_mod.check_numerical_anomaly(bad_t)
    finally:
        config.eager_mode = prev_eager
