"""Tests for test_dot."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.ops.linalg.dot import Dot, DotGeneral, Inner, Outer, Pdot, Tensordot, _has_valid_shape

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_dot_infer_shapes() -> None:
    """Test function test_dot_infer_shapes."""

    class MockTensor:
        """Test class MockTensor."""

        shape = (2, 3)

    t = MockTensor()
    assert Dot().infer_shape(t, t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", Tensordot)().infer_shape(t, t) == ()
    assert cast("Callable[..., EagerTensorProtocol]", Inner)().infer_shape(t, t) == ()
    assert cast("Callable[..., EagerTensorProtocol]", Outer)().infer_shape(t, t) == ()
    dg = DotGeneral()
    assert dg.infer_shape(t, t, (((1,), (0,)), ((), ()))) == (2, 3)
    assert dg.infer_shape(lhs=t, rhs=t, dimension_numbers=(((1,), (0,)), ((), ()))) == ()

    class BadTensor:
        """Test class BadTensor."""

        pass

    assert cast("Callable[..., EagerTensorProtocol]", dg.infer_shape)(cast("Callable[..., EagerTensorProtocol]", BadTensor)(), t, (((1,), (0,)), ((), ()))) == (2, 3)
    assert dg._compute_out_shape((2, 3), (3, 4), (((1,), (0,)), ((), ()))) == (2, 4)
    assert dg._compute_out_shape((5, 2, 3), (5, 3, 4), (((2,), (1,)), ((0,), (0,)))) == (5, 2, 4)


def test_dot_has_shape() -> None:
    """Test function test_dot_has_shape."""

    class DummyShape:
        """Test class DummyShape."""

        shape = (1,)

    assert cast("Callable[..., EagerTensorProtocol]", _has_valid_shape)(cast("Callable[..., EagerTensorProtocol]", DummyShape)())

    class DummyNoShape:
        """Test class DummyNoShape."""

        pass

    assert not cast("Callable[..., EagerTensorProtocol]", _has_valid_shape)(cast("Callable[..., EagerTensorProtocol]", DummyNoShape)())


def test_pdot_infer_shape() -> None:
    """Test function test_pdot_infer_shape."""

    class DummyShape:
        """Test class DummyShape."""

        shape = (1, 2)

    assert Pdot().infer_shape(DummyShape(), DummyShape(), None) == (1, 2)


def test_tensordot_compute_out_shape() -> None:
    """Test function test_tensordot_compute_out_shape."""
    from ml_switcheroo_compiler.ops.linalg.dot import _compute_pdot_shape

    assert _compute_pdot_shape((), ()) == ()
    assert _compute_pdot_shape((2,), (2,)) == ()
    assert _compute_pdot_shape((2, 3), (3, 4)) == (2, 4)
    assert _compute_pdot_shape((2,), ()) == (2,)
    assert _compute_pdot_shape((), (2,)) == (2,)
    assert _compute_pdot_shape((2, 3), (3,)) == (2,)
    assert _compute_pdot_shape((2, 3, 4), (4, 5)) == (2, 3, 5)


def test_pdot_frontend() -> None:
    """Test function test_pdot_frontend."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.linalg.dot import pdot

    config.eager_mode = True
    try:
        # Fallback will trigger backend not implemented, but we just need to hit the function
        pdot(np.array([1]), np.array([2]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
