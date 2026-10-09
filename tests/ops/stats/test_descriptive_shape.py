"""Test Descriptive shape inference."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.stats.descriptive import Descriptive


def test_descriptive_shape() -> None:
    """Test Descriptive shape inference."""
    op = Descriptive()

    a = type("MockTensor", (), {"shape": (4, 5, 6)})()

    # Default axis=None, keepdims=False
    assert op.infer_shape(a) == ((), (), (), ())

    # axis=None, keepdims=True
    assert op.infer_shape(a, keepdims=True) == ((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1))

    # axis=(0, 2), keepdims=False
    assert op.infer_shape(a, axis=(0, 2)) == ((5,), (5,), (5,), (5,))

    # axis=(0, 2), keepdims=True
    assert op.infer_shape(a, axis=(0, 2), keepdims=True) == ((1, 5, 1), (1, 5, 1), (1, 5, 1), (1, 5, 1))
