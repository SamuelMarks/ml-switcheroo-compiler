"""Tests for test_logical."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.ops.unary.logical import Ediff1d, Iscomplexobj, Isin, Isrealobj, Issubdtype, Packbits, Unpackbits, iscomplex, iscomplexobj, isin, isreal, isrealobj, issubdtype, population_count, reduce_precision

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


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
    assert cast("EagerCallableProtocol", Isin)().infer_shape(cast("EagerCallableProtocol", Dummy)(), cast("EagerCallableProtocol", Dummy)()) == (2, 3)


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
    assert cast("EagerCallableProtocol", Iscomplexobj)().infer_shape(cast("EagerCallableProtocol", Dummy)()) == ()
    assert cast("EagerCallableProtocol", Isrealobj)().infer_shape(cast("EagerCallableProtocol", Dummy)()) == ()
    assert Issubdtype().infer_shape() == ()
    op = Ediff1d()
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,))) == (4,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_begin=cast("EagerCallableProtocol", Dummy)(())) == (5,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_begin=cast("EagerCallableProtocol", Dummy)((2,))) == (6,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_end=cast("EagerCallableProtocol", Dummy)(())) == (5,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_end=cast("EagerCallableProtocol", Dummy)((2,))) == (6,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_begin=cast("EagerCallableProtocol", Dummy)((2,)), to_end=cast("EagerCallableProtocol", Dummy)((3,))) == (9,)
    from ml_switcheroo_compiler.ir.shape_system import SymVar

    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((None,))) == (cast("EagerCallableProtocol", SymVar)("diff_len"),)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_begin=cast("EagerCallableProtocol", Dummy)((None,))) == (cast("EagerCallableProtocol", SymVar)("diff_len"),)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_end=cast("EagerCallableProtocol", Dummy)((None,))) == (cast("EagerCallableProtocol", SymVar)("diff_len"),)
    assert op.infer_shape(None) == (0,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_begin=[1, 2]) == (6,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_end=(10,)) == (5,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5,)), to_begin=None) == (4,)
    assert cast("EagerCallableProtocol", op.infer_shape)(cast("EagerCallableProtocol", Dummy)((5, "str_dim")), to_begin=cast("EagerCallableProtocol", Dummy)(())) == (5,)


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
            cast("EagerCallableProtocol", func)(*args, **kwargs)

    cast("EagerCallableProtocol", safe_call)(population_count, cast("EagerCallableProtocol", Dummy)((2,)))
    cast("EagerCallableProtocol", safe_call)(isin, cast("EagerCallableProtocol", Dummy)((2,)), cast("EagerCallableProtocol", Dummy)((2,)))
    cast("EagerCallableProtocol", safe_call)(iscomplex, cast("EagerCallableProtocol", Dummy)((2,)))
    cast("EagerCallableProtocol", safe_call)(iscomplexobj, cast("EagerCallableProtocol", Dummy)((2,)))
    cast("EagerCallableProtocol", safe_call)(isreal, cast("EagerCallableProtocol", Dummy)((2,)))
    cast("EagerCallableProtocol", safe_call)(isrealobj, cast("EagerCallableProtocol", Dummy)((2,)))
    cast("EagerCallableProtocol", safe_call)(issubdtype, cast("EagerCallableProtocol", Dummy)((2,)), cast("EagerCallableProtocol", Dummy)(()))
    cast("EagerCallableProtocol", safe_call)(reduce_precision, cast("EagerCallableProtocol", Dummy)((2,)))


def test_logical_infer_shape_packbits() -> None:
    """Test function test_logical_infer_shape_packbits."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    op1 = Packbits()
    assert cast("EagerCallableProtocol", op1.infer_shape)(cast("EagerCallableProtocol", Dummy)((2,)), cast("EagerCallableProtocol", Dummy)((3, 2))) == (3, 2)
    op2 = Unpackbits()
    assert cast("EagerCallableProtocol", op2.infer_shape)(cast("EagerCallableProtocol", Dummy)((2,)), cast("EagerCallableProtocol", Dummy)((3, 2))) == (3, 2)


def test_logical_infer_shape_isin() -> None:
    """Test function test_logical_infer_shape_isin."""
    assert Isin().infer_shape() == ()


def test_logical_infer_shape_packbits_no_shapes() -> None:
    """Test function test_logical_infer_shape_packbits_no_shapes."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    class NoShape:
        """Test class NoShape."""

        pass

    assert cast("EagerCallableProtocol", Packbits)().infer_shape(cast("EagerCallableProtocol", NoShape)()) == ()
    assert cast("EagerCallableProtocol", Unpackbits)().infer_shape(cast("EagerCallableProtocol", NoShape)()) == ()
