"""Test Distributions shape inference."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.stats.descriptive import Distributions


def test_distributions_shape() -> None:
    """Test Distributions shape inference."""
    op = Distributions()

    a = type("MockTensor", (), {"shape": (4, 5, 6)})()

    # Default sample_shape=()
    assert op.infer_shape(a) == (4, 5, 6)

    # sample_shape=10
    assert op.infer_shape(a, sample_shape=10) == (10, 4, 5, 6)

    # sample_shape=(10, 20)
    assert op.infer_shape(a, sample_shape=(10, 20)) == (10, 20, 4, 5, 6)
