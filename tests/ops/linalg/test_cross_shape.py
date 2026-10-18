"""Test Cross shape inference."""

from __future__ import annotations

import pytest

from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import Cross


def test_cross_shape() -> None:
    """Test Cross shape inference."""
    op = Cross()

    a = type("MockTensor", (), {"shape": (4, 5, 3)})()
    b = type("MockTensor", (), {"shape": (4, 1, 3)})()

    assert op.infer_shape(a, b) == (4, 5, 3)

    a2 = type("MockTensor", (), {"shape": (3, 4, 5)})()
    b2 = type("MockTensor", (), {"shape": (3, 4, 1)})()

    assert op.infer_shape(a2, b2, axis=0) == (3, 4, 5)

    # Error cases
    a3 = type("MockTensor", (), {"shape": (4, 5, 4)})()
    b3 = type("MockTensor", (), {"shape": (4, 5, 4)})()
    with pytest.raises(ValueError):
        op.infer_shape(a3, b3)
