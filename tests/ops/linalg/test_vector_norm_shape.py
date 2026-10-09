"""Test VectorNorm shape inference."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.linalg.solvers import VectorNorm


def test_vector_norm_shape() -> None:
    """Test VectorNorm shape inference."""
    op = VectorNorm()

    a = type("MockTensor", (), {"shape": (4, 5, 6)})()

    # Default dim=-1, keepdims=False
    assert op.infer_shape(a) == (4, 5)

    # keepdims=True
    assert op.infer_shape(a, keepdims=True) == (4, 5, 1)

    # dim=(0, 2)
    assert op.infer_shape(a, dim=(0, 2)) == (5,)
