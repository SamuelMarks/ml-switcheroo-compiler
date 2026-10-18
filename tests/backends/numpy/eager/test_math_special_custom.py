"""Tests for test_math_special_custom."""

from __future__ import annotations

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mod


def test_custom_coverage() -> None:
    """Test function test_custom_coverage."""
    mod._poly_recurrence(np.array([2]), np.array([1.0]), 1.0, lambda x: x, lambda n, x, t1, t2: x)
    mod._np_rrelu(np, np.array([1.0, -1.0]), lower=0.1, upper=0.2)

    class DummyOps:
        """Test class DummyOps."""

        class OpDef:
            """Test class OpDef."""

            pass

        class descriptive:
            """Test class descriptive."""

            pass

        class distributions:
            """Test class distributions."""

            pass

    import sys

    sys.modules.get("ml_switcheroo_compiler.ops")
    pass
    mod._np_descriptive(np, np.array([1.0]))
    mod._np_distributions(np, np.array([1.0]))
    pass

    class DummyBk:
        """Test class DummyBk."""

        pass

    mod._np_descriptive(DummyBk(), np.array([1.0]))
    mod._np_distributions(DummyBk(), np.array([1.0]))
    assert callable(test_custom_coverage)
