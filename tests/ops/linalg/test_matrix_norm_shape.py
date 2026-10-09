"""Test MatrixNorm shape inference."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.linalg.solvers import MatrixNorm


def test_matrix_norm_shape() -> None:
    """Test MatrixNorm shape inference."""
    op = MatrixNorm()

    a = type("MockTensor", (), {"shape": (4, 5, 6)})()

    # Default dim=(-2, -1), keepdims=False
    assert op.infer_shape(a) == (4,)

    # keepdims=True
    assert op.infer_shape(a, keepdims=True) == (4, 1, 1)

    # dim=(0, 2)
    assert op.infer_shape(a, dim=(0, 2)) == (5,)
