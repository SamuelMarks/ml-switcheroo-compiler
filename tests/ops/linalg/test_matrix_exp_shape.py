"""Test MatrixExponential shape inference."""

from __future__ import annotations

import pytest

from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import MatrixExponential


def test_matrix_exponential_shape() -> None:
    """Test MatrixExponential shape inference."""
    op = MatrixExponential()

    a = type("MockTensor", (), {"shape": (4, 5, 5)})()
    assert op.infer_shape(a) == (4, 5, 5)

    # Error cases
    a2 = type("MockTensor", (), {"shape": (4, 5, 6)})()
    with pytest.raises(ValueError):
        op.infer_shape(a2)

    a3 = type("MockTensor", (), {"shape": (5,)})()
    with pytest.raises(ValueError):
        op.infer_shape(a3)
