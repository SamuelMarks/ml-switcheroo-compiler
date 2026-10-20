"""Tests for test_logical."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.ops.unary.logical import Ediff1d, Iscomplexobj, Isin, Isrealobj, Issubdtype, Packbits, Unpackbits, iscomplex, iscomplexobj, isin, isreal, isrealobj, issubdtype, population_count, reduce_precision

if typing.TYPE_CHECKING:
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
    assert typing.cast("EagerCallableProtocol", Isin)().infer_shape(typing.cast("EagerCallableProtocol", Dummy)(), typing.cast("EagerCallableProtocol", Dummy)()) == (2, 3)


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
    assert typing.cast("EagerCallableProtocol", Iscomplexobj)().infer_shape(typing.cast("EagerCallableProtocol", Dummy)()) == ()
    assert typing.cast("EagerCallableProtocol", Isrealobj)().infer_shape(typing.cast("EagerCallableProtocol", Dummy)()) == ()
    assert Issubdtype().infer_shape() == ()
    op = Ediff1d()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,))) == (4,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_begin=typing.cast("EagerCallableProtocol", Dummy)(())) == (5,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_begin=typing.cast("EagerCallableProtocol", Dummy)((2,))) == (6,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_end=typing.cast("EagerCallableProtocol", Dummy)(())) == (5,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_end=typing.cast("EagerCallableProtocol", Dummy)((2,))) == (6,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_begin=typing.cast("EagerCallableProtocol", Dummy)((2,)), to_end=typing.cast("EagerCallableProtocol", Dummy)((3,))) == (9,)
    from ml_switcheroo_compiler.ir.shape_system import SymVar

    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((None,))) == (typing.cast("EagerCallableProtocol", SymVar)("diff_len"),)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_begin=typing.cast("EagerCallableProtocol", Dummy)((None,))) == (typing.cast("EagerCallableProtocol", SymVar)("diff_len"),)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_end=typing.cast("EagerCallableProtocol", Dummy)((None,))) == (typing.cast("EagerCallableProtocol", SymVar)("diff_len"),)
    assert op.infer_shape(None) == (0,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_begin=[1, 2]) == (6,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_end=(10,)) == (5,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), to_begin=None) == (4,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5, "str_dim")), to_begin=typing.cast("EagerCallableProtocol", Dummy)(())) == (5,)


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
            typing.cast("EagerCallableProtocol", func)(*args, **kwargs)

    typing.cast("EagerCallableProtocol", safe_call)(population_count, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(isin, typing.cast("EagerCallableProtocol", Dummy)((2,)), typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(iscomplex, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(iscomplexobj, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(isreal, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(isrealobj, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(issubdtype, typing.cast("EagerCallableProtocol", Dummy)((2,)), typing.cast("EagerCallableProtocol", Dummy)(()))
    typing.cast("EagerCallableProtocol", safe_call)(reduce_precision, typing.cast("EagerCallableProtocol", Dummy)((2,)))


def test_logical_infer_shape_packbits() -> None:
    """Test function test_logical_infer_shape_packbits."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    op1 = Packbits()
    assert typing.cast("EagerCallableProtocol", op1.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2,)), typing.cast("EagerCallableProtocol", Dummy)((3, 2))) == (3, 2)
    op2 = Unpackbits()
    assert typing.cast("EagerCallableProtocol", op2.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2,)), typing.cast("EagerCallableProtocol", Dummy)((3, 2))) == (3, 2)


def test_logical_infer_shape_isin() -> None:
    """Test function test_logical_infer_shape_isin."""
    assert Isin().infer_shape() == ()


def test_logical_infer_shape_packbits_no_shapes() -> None:
    """Test function test_logical_infer_shape_packbits_no_shapes."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    class NoShape:
        """Test class NoShape."""

        pass

    assert typing.cast("EagerCallableProtocol", Packbits)().infer_shape(typing.cast("EagerCallableProtocol", NoShape)()) == ()
    assert typing.cast("EagerCallableProtocol", Unpackbits)().infer_shape(typing.cast("EagerCallableProtocol", NoShape)()) == ()
