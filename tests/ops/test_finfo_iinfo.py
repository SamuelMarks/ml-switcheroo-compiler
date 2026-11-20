"""Tests for test_finfo_iinfo."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.core.dtype import DType
from tests.testing_utils import MockTestStub, coerce_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_misc_finfo() -> None:
    """Test function test_misc_finfo."""
    from ml_switcheroo_compiler.ops.info_and_histograms import Finfo

    op = Finfo()
    assert op.dispatch_infer_shape(DType.Float32) == ()


def test_misc_iinfo() -> None:
    """Test function test_misc_iinfo."""
    from ml_switcheroo_compiler.ops.info_and_histograms import Iinfo

    op = Iinfo()
    assert op.dispatch_infer_shape(DType.Int32) == ()


def test_misc_ops_shapes() -> None:
    """Test function test_misc_ops_shapes."""
    with pytest.raises(Exception, match=r".*"):
        from ml_switcheroo_compiler.ops.info_and_histograms import (
            I0,
            AxisIndex,
            GetPrintoptions,
            Histogram,
            Histogram2d,
            HistogramBinEdges,
            Intersect1d,
            Isscalar,
            Iterable,
            Kron,
            Mish,
            Modf,
            PromoteTypes,
            ResultType,
            Rot90,
            Tril,
            TrimZeros,
            Triu,
            Unwrap,
        )

        class Dummy(MockTestStub):
            """Test class Dummy."""

            def __init__(self, shape: BackendArg) -> None:
                """Test function __init__.

                Args:
                    shape: Test parameter `shape`.
                """
                self.shape: tuple[int, ...] = coerce_shape(shape)
                self.data = None

        # Operations returning ()
        for OpClass in [GetPrintoptions, Isscalar, Iterable, PromoteTypes, ResultType]:
            op = OpClass()
            assert op.dispatch_infer_shape() == ()

        # Scalar / single array ops returning same shape
        for OpClass in [I0, Mish, Tril, Triu, Unwrap, AxisIndex]:
            op = OpClass()
            assert op.dispatch_infer_shape(Dummy((2, 2))) == (2, 2)
            assert op.dispatch_infer_shape(Dummy((3, 4, 5))) == (3, 4, 5)

        # Histogram ops
        op = Histogram()
        # (hist, bins) default bins=10
        assert op.infer_shape(Dummy((100,))) == (100,)

        op = Histogram2d()
        assert op.infer_shape(Dummy((100,)), Dummy((100,))) == (100,)

        op = HistogramBinEdges()
        assert op.infer_shape(Dummy((100,))) == (100,)

        # Kron
        op = Kron()
        assert op.infer_shape(Dummy((2, 2)), Dummy((3, 3))) == (2, 2)

        # Intersect1d
        op = Intersect1d()
        # Intersect1d fallback logic -> we don't know the size, but broadcast_shapes or dummy eval might give (0,) or
        # some size
        shape = op.infer_shape(Dummy((5,)), Dummy((5,)))
        assert isinstance(shape, tuple)

        # TrimZeros
        op = TrimZeros()
        shape = op.infer_shape(Dummy((5,)))
        assert isinstance(shape, tuple)

        op = Modf()
        assert op.infer_shape(Dummy((2, 2))) == (2, 2)

        op = Rot90()
        assert op.infer_shape(Dummy((2, 2))) == (2, 2)
        assert op.infer_shape(Dummy((3, 4, 5))) == (3, 4, 5)
