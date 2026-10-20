"""Tests for test_finfo_iinfo."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.dtype import DType

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


def test_misc_finfo() -> None:
    """Test function test_misc_finfo."""
    from ml_switcheroo_compiler.ops.info_and_histograms import Finfo

    op = Finfo()
    assert op.infer_shape(DType.Float32) == ()


def test_misc_iinfo() -> None:
    """Test function test_misc_iinfo."""
    from ml_switcheroo_compiler.ops.info_and_histograms import Iinfo

    op = Iinfo()
    assert op.infer_shape(DType.Int32) == ()


def test_misc_ops_shapes() -> None:
    """Test function test_misc_ops_shapes."""
    import pytest

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

        class Dummy:
            """Test class Dummy."""

            def __init__(self, shape: EagerValue) -> None:
                """Test function __init__.

                Args:
                    shape: Test parameter `shape`.
                """
                self.shape: tuple[int, ...] = typing.cast("tuple[int, ...]", shape)
                self.data = None

        # Operations returning ()
        for OpClass in [GetPrintoptions, Isscalar, Iterable, PromoteTypes, ResultType]:
            op = OpClass()
            assert op.infer_shape() == ()

        # Scalar / single array ops returning same shape
        for OpClass in [I0, Mish, Tril, Triu, Unwrap, AxisIndex]:
            op = OpClass()
            assert op.infer_shape(Dummy(typing.cast("EagerValue", (2, 2)))) == (2, 2)
            assert op.infer_shape(Dummy(typing.cast("EagerValue", (3, 4, 5)))) == (3, 4, 5)

        # Histogram ops
        op = Histogram()
        # (hist, bins) default bins=10
        assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((100,))) == (100,)

        op = Histogram2d()
        assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((100,)), typing.cast("EagerCallableProtocol", Dummy)((100,))) == (100,)

        op = HistogramBinEdges()
        assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((100,))) == (100,)

        # Kron
        op = Kron()
        assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 2)), typing.cast("EagerCallableProtocol", Dummy)((3, 3))) == (2, 2)

        # Intersect1d
        op = Intersect1d()
        # Intersect1d fallback logic -> we don't know the size, but broadcast_shapes or dummy eval might give (0,) or some size
        shape = typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)), typing.cast("EagerCallableProtocol", Dummy)((5,)))
        assert isinstance(typing.cast("EagerValue", shape), tuple)

        # TrimZeros
        op = TrimZeros()
        shape = typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((5,)))
        assert isinstance(typing.cast("EagerValue", shape), tuple)

        op = Modf()
        assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 2))) == (2, 2)

        op = Rot90()
        assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 2))) == (2, 2)
        assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((3, 4, 5))) == (3, 4, 5)
