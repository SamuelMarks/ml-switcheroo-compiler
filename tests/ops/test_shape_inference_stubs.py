"""Tests for operator shape inference stubs across operations."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class Dummy:
    """Dummy tensor-like object with a 2D shape."""

    shape = (2, 2)

    def __iter__(self) -> EagerValue:
        """Test function __iter__."""
        return typing.cast("EagerTensorProtocol", iter(self.shape))


def test_shape_misc_stubs() -> None:
    """Verify shape inference on miscellaneous shape ops."""
    from ml_switcheroo_compiler.ops.shape.pad_and_tile import (
        Flatnonzero,
        Lexsort,
        Nonzero,
        Percentile,
        Quantile,
        RavelMultiIndex,
        Repeat,
        Searchsorted,
        SortComplex,
        Tile,
        Unique,
    )

    for OpClass in [Flatnonzero, Lexsort, Nonzero, Percentile, Quantile, RavelMultiIndex, Repeat, Searchsorted, SortComplex, Tile, Unique]:
        op = OpClass()
        assert isinstance(op.infer_shape(Dummy()), tuple)
        assert isinstance(op.infer_shape(Dummy(), Dummy()), tuple)
        assert isinstance(op.infer_shape(), tuple)


def test_shape_slicing_stubs() -> None:
    """Verify shape inference on IndexInDim and UpdateSlice."""
    from ml_switcheroo_compiler.ops.shape.slicing import IndexInDim, UpdateSlice

    for OpClass in [IndexInDim, UpdateSlice]:
        op = OpClass()
        assert isinstance(op.infer_shape(Dummy()), tuple)
        assert isinstance(op.infer_shape(), tuple)


def test_creation_frontend_stubs() -> None:
    """Verify shape inference on creation frontend ops."""
    from ml_switcheroo_compiler.ops.creation.frontend_utils import FromDlpack, Frompyfunc, Geometric, Geomspace

    for OpClass in [Geometric, Geomspace, FromDlpack, Frompyfunc]:
        op = OpClass()
        assert isinstance(op.infer_shape(Dummy()), tuple)
        assert isinstance(op.infer_shape(Dummy(), Dummy(), Dummy()), tuple)
        assert isinstance(op.infer_shape(size=(1, 2)), tuple)
        assert isinstance(op.infer_shape(num=5), tuple)
        assert isinstance(op.infer_shape(), tuple)


def test_nn_activations_stubs() -> None:
    """Verify shape inference on neural network activation ops."""
    from ml_switcheroo_compiler.ops.nn.activations import HardSilu, HardSwish, Squareplus

    for OpClass in [HardSilu, HardSwish, Squareplus]:
        op = OpClass()
        assert op.infer_shape(Dummy()) == (2, 2)
        assert op.infer_shape() == ()


def test_distributed_stubs() -> None:
    """Verify shape inference on distributed collective ops."""
    from ml_switcheroo_compiler.ops.distributed_ops import Outfeed, Pmax, Pmin, Ppermute, Pshuffle, PsumScatter, Pswapaxes

    for OpClass in [Outfeed, Pshuffle, Pswapaxes, Ppermute, PsumScatter, Pmax, Pmin]:
        op = OpClass()
        assert isinstance(op.infer_shape(Dummy()), tuple)
        assert op.infer_shape() == ()


def test_random_stubs() -> None:
    """Verify shape inference on Rademacher op."""
    from ml_switcheroo_compiler.ops.random_ops.core import Rademacher

    op = Rademacher()
    assert isinstance(op.infer_shape(Dummy()), tuple)
    assert op.infer_shape(shape=(1, 2)) == (1, 2)
    assert op.infer_shape() == ()


def test_unary_logical_stubs() -> None:
    """Verify shape inference on unary logical ops."""
    from ml_switcheroo_compiler.ops.unary.logical import Ediff1d, Iscomplex, Iscomplexobj, Isin, Isreal, Isrealobj, Issubdtype

    for OpClass in [Iscomplex, Iscomplexobj, Isreal, Isrealobj, Issubdtype, Isin]:
        op = OpClass()
        assert isinstance(op.infer_shape(Dummy()), tuple)
        assert op.infer_shape() == ()
    ediff_op = Ediff1d()
    assert isinstance(ediff_op.infer_shape(Dummy()), tuple)
    assert ediff_op.infer_shape() == (0,)


def test_linalg_dot_stubs() -> None:
    """Verify shape inference on Pdot op."""
    from ml_switcheroo_compiler.ops.linalg.dot import Pdot

    op = Pdot()
    assert isinstance(op.infer_shape(Dummy(), Dummy()), tuple)
    with pytest.raises((ValueError, IndexError)):
        op.infer_shape()


def test_io_stubs() -> None:
    """Verify shape inference on I/O ops."""
    from ml_switcheroo_compiler.ops.io import Fromfile, Fromfunction, Fromiter, Fromstring

    for OpClass in [Fromfile, Fromstring, Fromiter, Fromfunction]:
        op = OpClass()
        assert isinstance(op.infer_shape(Dummy()), tuple)
        assert isinstance(op.infer_shape(Dummy(), Dummy()), tuple)
        assert isinstance(op.infer_shape(shape=(2, 2)), tuple)
        assert isinstance(op.infer_shape(), tuple)


def test_broadcast_to_empty_shape() -> None:
    """Verify BroadcastTo shape inference edge cases."""
    from ml_switcheroo_compiler.ops.distributed_ops import BroadcastTo

    class NoShape:
        """Test class NoShape."""

        pass

    op = BroadcastTo()
    assert op.infer_shape() == ()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(42) == ()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", NoShape)()) == ()
    assert op.infer_shape(shape=(5, 5)) == (5, 5)


def test_dot_and_pdot_infer_shape_edge_cases() -> None:
    """Verify Dot and Pdot shape inference across 0D, 1D, 2D, and ND inputs."""
    from ml_switcheroo_compiler.ops.linalg.dot import Dot, Pdot

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    for OpClass in (Dot, Pdot):
        op = OpClass()
        assert op.infer_shape(MockTensor((2,)), MockTensor((2,))) == ()
        assert op.infer_shape(MockTensor(()), MockTensor((2,))) == (2,)
        assert op.infer_shape(MockTensor((2,)), MockTensor(())) == (2,)
        assert op.infer_shape(MockTensor((2, 2)), MockTensor((2,))) == (2,)
        assert op.infer_shape(MockTensor((2, 3, 4)), MockTensor((4, 5))) == (2, 3, 5)


def test_ops_shape_inference_inputs_type_and_missing_op() -> None:
    """Test infer_shape with non-sequence inputs kwarg and non-existent op."""
    from ml_switcheroo_compiler.ops.shape_inference import infer_shape

    res1 = infer_shape("Add", inputs=12345)
    assert res1 == ()
    with patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=None):
        res2 = infer_shape("NonExistentOpXYZ_12345")
        assert res2 == ()
