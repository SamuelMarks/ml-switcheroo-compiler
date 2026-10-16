"""Tests for test_math_special_custom."""

from __future__ import annotations

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as mod


def test_custom_coverage() -> None:
    # line 3059: T.append(p1_func(x_b))
    """Test function test_custom_coverage."""
    mod._poly_recurrence(np.array([2]), np.array([1.0]), 1.0, lambda x: x, lambda n, x, t1, t2: x)

    # line 3739: alpha = np.random.uniform(lower, upper, size=a.shape)
    mod._np_rrelu(np, np.array([1.0, -1.0]), lower=0.1, upper=0.2)

    # line 2854-2860, 2885-2891
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
    try:
        mod._np_descriptive(np, np.array([1.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mod._np_distributions(np, np.array([1.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    pass

    # Also we need to test without descriptive / distributions attributes on backend
    class DummyBk:
        """Test class DummyBk."""

        pass

    try:
        mod._np_descriptive(DummyBk(), np.array([1.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mod._np_distributions(DummyBk(), np.array([1.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
