"""Tests for test_logical."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.ops.unary.logical import (
    Ediff1d,
    Iscomplexobj,
    Isin,
    Isrealobj,
    Issubdtype,
    Packbits,
    Unpackbits,
    iscomplex,
    iscomplexobj,
    isin,
    isreal,
    isrealobj,
    issubdtype,
    population_count,
    reduce_precision,
)
from tests.testing_utils import MockTestStub, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_logical_infer_shape() -> None:
    """Test function test_logical_infer_shape."""
    assert Packbits().infer_shape() == ()
    assert Unpackbits().infer_shape() == ()

    class Dummy(MockTestStub):
        """Test class Dummy."""

        shape = (2, 3)

    assert Packbits().infer_shape(Dummy()) == (2, 3)
    assert Unpackbits().infer_shape(Dummy()) == (2, 3)
    assert Packbits().infer_shape(Dummy(), Dummy()) == (2, 3)
    assert Unpackbits().infer_shape(Dummy(), Dummy()) == (2, 3)
    from ml_switcheroo_compiler.ops.unary.logical import Isin

    assert Isin().infer_shape() == ()
    assert Isin().infer_shape(Dummy(), Dummy()) == (2, 3)


class Dummy(MockTestStub):
    """Dummy tensor class for testing."""

    def __init__(self, shape: BackendArg = None) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        if shape is not None:
            self.shape = shape


def test_logical_infer_shape_additional() -> None:
    """Test function test_logical_infer_shape_additional."""
    assert Iscomplexobj().infer_shape(Dummy()) == ()
    assert Isrealobj().infer_shape(Dummy()) == ()
    assert Issubdtype().infer_shape() == ()
    op = Ediff1d()
    assert op.infer_shape(Dummy((5,))) == (4,)
    assert op.infer_shape(Dummy((5,)), to_begin=Dummy(())) == (5,)
    assert op.infer_shape(Dummy((5,)), to_begin=Dummy((2,))) == (6,)
    assert op.infer_shape(Dummy((5,)), to_end=Dummy(())) == (5,)
    assert op.infer_shape(Dummy((5,)), to_end=Dummy((2,))) == (6,)
    assert op.infer_shape(Dummy((5,)), to_begin=Dummy((2,)), to_end=Dummy((3,))) == (9,)
    from ml_switcheroo_compiler.ir.shape_system import SymVar

    assert op.infer_shape(Dummy((None,))) == (SymVar("diff_len"),)
    assert op.infer_shape(Dummy((5,)), to_begin=Dummy((None,))) == (SymVar("diff_len"),)
    assert op.infer_shape(Dummy((5,)), to_end=Dummy((None,))) == (SymVar("diff_len"),)
    assert op.infer_shape(None) == (0,)
    assert op.infer_shape(Dummy((5,)), to_begin=[1, 2]) == (6,)
    assert op.infer_shape(Dummy((5,)), to_end=(10,)) == (5,)
    assert op.infer_shape(Dummy((5,)), to_begin=None) == (4,)
    assert op.infer_shape(Dummy((5, "str_dim")), to_begin=Dummy(())) == (5,)


def test_logical_dispatch() -> None:
    """Test function test_logical_dispatch."""

    def safe_call(func: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Test function safe_call.

        Args:
            func: Test parameter `func`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        with pytest.raises(RuntimeError):
            require_callable(func)(*args, **kwargs)

    safe_call(population_count, Dummy((2,)))
    safe_call(isin, Dummy((2,)), Dummy((2,)))
    safe_call(iscomplex, Dummy((2,)))
    safe_call(iscomplexobj, Dummy((2,)))
    safe_call(isreal, Dummy((2,)))
    safe_call(isrealobj, Dummy((2,)))
    safe_call(issubdtype, Dummy((2,)), Dummy(()))
    safe_call(reduce_precision, Dummy((2,)))


def test_logical_infer_shape_packbits() -> None:
    """Test function test_logical_infer_shape_packbits."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    op1 = Packbits()
    assert op1.infer_shape(Dummy((2,)), Dummy((3, 2))) == (3, 2)
    op2 = Unpackbits()
    assert op2.infer_shape(Dummy((2,)), Dummy((3, 2))) == (3, 2)


def test_logical_infer_shape_isin() -> None:
    """Test function test_logical_infer_shape_isin."""
    assert Isin().infer_shape() == ()


def test_logical_infer_shape_packbits_no_shapes() -> None:
    """Test function test_logical_infer_shape_packbits_no_shapes."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    class NoShape(MockTestStub):
        """Test class NoShape."""

        pass

    assert Packbits().infer_shape(NoShape()) == ()
    assert Unpackbits().infer_shape(NoShape()) == ()
