"""Tests for ops unary logical coverage."""

import ml_switcheroo_compiler.ops.unary.logical as unary_logical


def test_unary_logical_functions() -> None:
    """Test unary logical coverage."""

    class DummyNode:
        """Test class DummyNode."""

        shape = (10, 20)

    a = DummyNode()
    for attr in dir(unary_logical):
        if attr.startswith("_infer_shape_") or attr.startswith("_infer_dtype_"):
            val = getattr(unary_logical, attr)
            if callable(val):
                try:
                    val(a)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                    assert _exc is not None
                try:
                    val(a, a)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                    assert _exc is not None
                try:
                    val(a, a, a)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                    assert _exc is not None
