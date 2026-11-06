"""Tests for test_logical."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.ops.unary.logical import Ediff1d, Iscomplexobj, Isin, Isrealobj, Issubdtype, Packbits, Unpackbits, iscomplex, iscomplexobj, isin, isreal, isrealobj, issubdtype, population_count, reduce_precision
from tests.testing_utils import ensure_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_logical_infer_shape() -> None:
    """Test function test_logical_infer_shape."""
    assert Packbits().infer_shape() == ()
    assert Unpackbits().infer_shape() == ()

    class Dummy:
        """Test class Dummy."""

        shape = (2, 3)

    assert Packbits().infer_shape(Dummy()) == (2, 3)
    assert Unpackbits().infer_shape(Dummy()) == (2, 3)
    assert Packbits().infer_shape(Dummy(), Dummy()) == (2, 3)
    assert Unpackbits().infer_shape(Dummy(), Dummy()) == (2, 3)
    from ml_switcheroo_compiler.ops.unary.logical import Isin

    assert Isin().infer_shape() == ()
    assert ensure_callable(Isin)().infer_shape(ensure_callable(Dummy)(), ensure_callable(Dummy)()) == (2, 3)


class Dummy:
    """Dummy tensor class for testing."""

    def __init__(self, shape: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        if shape is not None:
            self.shape = shape


def test_logical_infer_shape_additional() -> None:
    """Test function test_logical_infer_shape_additional."""
    assert ensure_callable(Iscomplexobj)().infer_shape(ensure_callable(Dummy)()) == ()
    assert ensure_callable(Isrealobj)().infer_shape(ensure_callable(Dummy)()) == ()
    assert Issubdtype().infer_shape() == ()
    op = Ediff1d()
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,))) == (4,)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_begin=ensure_callable(Dummy)(())) == (5,)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_begin=ensure_callable(Dummy)((2,))) == (6,)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_end=ensure_callable(Dummy)(())) == (5,)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_end=ensure_callable(Dummy)((2,))) == (6,)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_begin=ensure_callable(Dummy)((2,)), to_end=ensure_callable(Dummy)((3,))) == (9,)
    from ml_switcheroo_compiler.ir.shape_system import SymVar

    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((None,))) == (ensure_callable(SymVar)("diff_len"),)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_begin=ensure_callable(Dummy)((None,))) == (ensure_callable(SymVar)("diff_len"),)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_end=ensure_callable(Dummy)((None,))) == (ensure_callable(SymVar)("diff_len"),)
    assert op.infer_shape(None) == (0,)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_begin=[1, 2]) == (6,)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_end=(10,)) == (5,)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5,)), to_begin=None) == (4,)
    assert ensure_callable(op.infer_shape)(ensure_callable(Dummy)((5, "str_dim")), to_begin=ensure_callable(Dummy)(())) == (5,)


def test_logical_dispatch() -> None:
    """Test function test_logical_dispatch."""

    def safe_call(func: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function safe_call.

        Args:
            func: Test parameter `func`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        with pytest.raises(RuntimeError):
            ensure_callable(func)(*args, **kwargs)

    ensure_callable(safe_call)(population_count, ensure_callable(Dummy)((2,)))
    ensure_callable(safe_call)(isin, ensure_callable(Dummy)((2,)), ensure_callable(Dummy)((2,)))
    ensure_callable(safe_call)(iscomplex, ensure_callable(Dummy)((2,)))
    ensure_callable(safe_call)(iscomplexobj, ensure_callable(Dummy)((2,)))
    ensure_callable(safe_call)(isreal, ensure_callable(Dummy)((2,)))
    ensure_callable(safe_call)(isrealobj, ensure_callable(Dummy)((2,)))
    ensure_callable(safe_call)(issubdtype, ensure_callable(Dummy)((2,)), ensure_callable(Dummy)(()))
    ensure_callable(safe_call)(reduce_precision, ensure_callable(Dummy)((2,)))


def test_logical_infer_shape_packbits() -> None:
    """Test function test_logical_infer_shape_packbits."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    op1 = Packbits()
    assert ensure_callable(op1.infer_shape)(ensure_callable(Dummy)((2,)), ensure_callable(Dummy)((3, 2))) == (3, 2)
    op2 = Unpackbits()
    assert ensure_callable(op2.infer_shape)(ensure_callable(Dummy)((2,)), ensure_callable(Dummy)((3, 2))) == (3, 2)


def test_logical_infer_shape_isin() -> None:
    """Test function test_logical_infer_shape_isin."""
    assert Isin().infer_shape() == ()


def test_logical_infer_shape_packbits_no_shapes() -> None:
    """Test function test_logical_infer_shape_packbits_no_shapes."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    class NoShape:
        """Test class NoShape."""

        pass

    assert ensure_callable(Packbits)().infer_shape(ensure_callable(NoShape)()) == ()
    assert ensure_callable(Unpackbits)().infer_shape(ensure_callable(NoShape)()) == ()
