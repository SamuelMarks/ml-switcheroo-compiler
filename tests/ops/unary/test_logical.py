"""Tests for test_logical."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.ops.unary.logical import Ediff1d, Iscomplexobj, Isin, Isrealobj, Issubdtype, Packbits, Unpackbits, iscomplex, iscomplexobj, isin, isreal, isrealobj, issubdtype, population_count, reduce_precision
from tests.testing_utils import as_callable

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
    assert as_callable(Isin)().infer_shape(as_callable(Dummy)(), as_callable(Dummy)()) == (2, 3)


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
    assert as_callable(Iscomplexobj)().infer_shape(as_callable(Dummy)()) == ()
    assert as_callable(Isrealobj)().infer_shape(as_callable(Dummy)()) == ()
    assert Issubdtype().infer_shape() == ()
    op = Ediff1d()
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,))) == (4,)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_begin=as_callable(Dummy)(())) == (5,)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_begin=as_callable(Dummy)((2,))) == (6,)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_end=as_callable(Dummy)(())) == (5,)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_end=as_callable(Dummy)((2,))) == (6,)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_begin=as_callable(Dummy)((2,)), to_end=as_callable(Dummy)((3,))) == (9,)
    from ml_switcheroo_compiler.ir.shape_system import SymVar

    assert as_callable(op.infer_shape)(as_callable(Dummy)((None,))) == (as_callable(SymVar)("diff_len"),)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_begin=as_callable(Dummy)((None,))) == (as_callable(SymVar)("diff_len"),)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_end=as_callable(Dummy)((None,))) == (as_callable(SymVar)("diff_len"),)
    assert op.infer_shape(None) == (0,)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_begin=[1, 2]) == (6,)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_end=(10,)) == (5,)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5,)), to_begin=None) == (4,)
    assert as_callable(op.infer_shape)(as_callable(Dummy)((5, "str_dim")), to_begin=as_callable(Dummy)(())) == (5,)


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
            as_callable(func)(*args, **kwargs)

    as_callable(safe_call)(population_count, as_callable(Dummy)((2,)))
    as_callable(safe_call)(isin, as_callable(Dummy)((2,)), as_callable(Dummy)((2,)))
    as_callable(safe_call)(iscomplex, as_callable(Dummy)((2,)))
    as_callable(safe_call)(iscomplexobj, as_callable(Dummy)((2,)))
    as_callable(safe_call)(isreal, as_callable(Dummy)((2,)))
    as_callable(safe_call)(isrealobj, as_callable(Dummy)((2,)))
    as_callable(safe_call)(issubdtype, as_callable(Dummy)((2,)), as_callable(Dummy)(()))
    as_callable(safe_call)(reduce_precision, as_callable(Dummy)((2,)))


def test_logical_infer_shape_packbits() -> None:
    """Test function test_logical_infer_shape_packbits."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    op1 = Packbits()
    assert as_callable(op1.infer_shape)(as_callable(Dummy)((2,)), as_callable(Dummy)((3, 2))) == (3, 2)
    op2 = Unpackbits()
    assert as_callable(op2.infer_shape)(as_callable(Dummy)((2,)), as_callable(Dummy)((3, 2))) == (3, 2)


def test_logical_infer_shape_isin() -> None:
    """Test function test_logical_infer_shape_isin."""
    assert Isin().infer_shape() == ()


def test_logical_infer_shape_packbits_no_shapes() -> None:
    """Test function test_logical_infer_shape_packbits_no_shapes."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    class NoShape:
        """Test class NoShape."""

        pass

    assert as_callable(Packbits)().infer_shape(as_callable(NoShape)()) == ()
    assert as_callable(Unpackbits)().infer_shape(as_callable(NoShape)()) == ()
