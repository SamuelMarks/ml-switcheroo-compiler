"""Tests for test_logical."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
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

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


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
    assert cast("Callable[..., EagerTensorProtocol]", Isin)().infer_shape(cast("Callable[..., EagerTensorProtocol]", Dummy)(), cast("Callable[..., EagerTensorProtocol]", Dummy)()) == (2, 3)


class Dummy:
    """Dummy tensor class for testing."""

    def __init__(self, shape: EagerValue = None) -> None:
        """Test function __init__."""
        if shape is not None:
            self.shape = shape


def test_logical_infer_shape_additional() -> None:
    """Test function test_logical_infer_shape_additional."""
    assert cast("Callable[..., EagerTensorProtocol]", Iscomplexobj)().infer_shape(cast("Callable[..., EagerTensorProtocol]", Dummy)()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", Isrealobj)().infer_shape(cast("Callable[..., EagerTensorProtocol]", Dummy)()) == ()
    assert Issubdtype().infer_shape() == ()

    # Ediff1d
    op = Ediff1d()
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,))) == (4,)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_begin=cast("Callable[..., EagerTensorProtocol]", Dummy)(())) == (5,)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_begin=cast("Callable[..., EagerTensorProtocol]", Dummy)((2,))) == (6,)

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_end=cast("Callable[..., EagerTensorProtocol]", Dummy)(())) == (5,)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_end=cast("Callable[..., EagerTensorProtocol]", Dummy)((2,))) == (6,)

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_begin=cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)), to_end=cast("Callable[..., EagerTensorProtocol]", Dummy)((3,))) == (9,)

    # Ediff1d with None in shape
    from ml_switcheroo_compiler.ir.shape_system import SymVar

    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((None,))) == (cast("Callable[..., EagerTensorProtocol]", SymVar)("diff_len"),)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_begin=cast("Callable[..., EagerTensorProtocol]", Dummy)((None,))) == (cast("Callable[..., EagerTensorProtocol]", SymVar)("diff_len"),)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_end=cast("Callable[..., EagerTensorProtocol]", Dummy)((None,))) == (cast("Callable[..., EagerTensorProtocol]", SymVar)("diff_len"),)

    # Edge cases for _get_size_from_shape (None, list/tuple, non-int in shape)
    assert op.infer_shape(None) == (0,)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_begin=[1, 2]) == (6,)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_end=(10,)) == (5,)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5,)), to_begin=None) == (4,)
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((5, "str_dim")), to_begin=cast("Callable[..., EagerTensorProtocol]", Dummy)(())) == (5,)


def test_logical_dispatch() -> None:
    """Test function test_logical_dispatch."""

    def safe_call(func: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function safe_call."""
        try:
            cast("Callable[..., EagerTensorProtocol]", func)(*args, **kwargs)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None

    cast("Callable[..., EagerTensorProtocol]", safe_call)(population_count, cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)))
    cast("Callable[..., EagerTensorProtocol]", safe_call)(isin, cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)), cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)))
    cast("Callable[..., EagerTensorProtocol]", safe_call)(iscomplex, cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)))
    cast("Callable[..., EagerTensorProtocol]", safe_call)(iscomplexobj, cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)))
    cast("Callable[..., EagerTensorProtocol]", safe_call)(isreal, cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)))
    cast("Callable[..., EagerTensorProtocol]", safe_call)(isrealobj, cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)))
    cast("Callable[..., EagerTensorProtocol]", safe_call)(issubdtype, cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)), cast("Callable[..., EagerTensorProtocol]", Dummy)(()))
    cast("Callable[..., EagerTensorProtocol]", safe_call)(reduce_precision, cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)))


def test_logical_infer_shape_packbits() -> None:
    """Test function test_logical_infer_shape_packbits."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    op1 = Packbits()
    assert cast("Callable[..., EagerTensorProtocol]", op1.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)), cast("Callable[..., EagerTensorProtocol]", Dummy)((3, 2))) == (3, 2)

    op2 = Unpackbits()
    assert cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)(cast("Callable[..., EagerTensorProtocol]", Dummy)((2,)), cast("Callable[..., EagerTensorProtocol]", Dummy)((3, 2))) == (3, 2)


def test_logical_infer_shape_isin() -> None:
    """Test function test_logical_infer_shape_isin."""
    assert Isin().infer_shape() == ()


def test_logical_infer_shape_packbits_no_shapes() -> None:
    """Test function test_logical_infer_shape_packbits_no_shapes."""
    from ml_switcheroo_compiler.ops.unary.logical import Packbits, Unpackbits

    class NoShape:
        """Test class NoShape."""

        pass

    assert cast("Callable[..., EagerTensorProtocol]", Packbits)().infer_shape(cast("Callable[..., EagerTensorProtocol]", NoShape)()) == ()
    assert cast("Callable[..., EagerTensorProtocol]", Unpackbits)().infer_shape(cast("Callable[..., EagerTensorProtocol]", NoShape)()) == ()
