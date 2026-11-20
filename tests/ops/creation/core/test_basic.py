"""Tests for test_basic."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.ops.creation.basic import (
    Arange,
    Bartlett,
    Blackman,
    ConstantOfShape,
    CreationOp,
    Frombuffer,
    Full,
    Hamming,
    Hanning,
    Kaiser,
    Logspace,
    ManualSeed,
    Rand,
    Randint,
    Range,
    TrilIndices,
    TrilIndicesFrom,
    TriuIndices,
    TriuIndicesFrom,
)
from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class DummyItem(MockTestStub):
    """Test class DummyItem."""

    def item(self) -> BackendArg:
        """Test function item.

        Returns:
            BackendArg: Computed output value from `item`.
        """
        return 5


def test_creation_basic() -> None:
    """Test function test_creation_basic."""
    c = CreationOp()
    assert c.infer_shape((2, 2)) == (2, 2)

    f = Full()
    assert f.infer_shape((2, 2), 0.0) == (2, 2)

    assert Arange().infer_shape(10) is None

    r = Rand()
    assert r.infer_shape(size=(2, 2)) == (2, 2)
    assert r.infer_shape((2, 2)) == (2, 2)
    assert r.infer_shape(2, 2) == (2, 2)

    ri = Randint()
    assert ri.infer_shape(0, 10, size=(2, 2)) == (2, 2)
    assert ri.infer_shape(0, 10, (2, 2)) == (2, 2)
    assert ri.infer_shape(0, 10) == ()

    assert ManualSeed().infer_shape(42) == ()

    assert ConstantOfShape().infer_shape((2, 2)) == (2, 2)
    assert ConstantOfShape().infer_shape() == ()

    assert Range().infer_shape((2, 2)) == (2, 2)
    assert Range().infer_shape() == ()

    for cls in [Blackman, Bartlett, Hamming, Hanning, Kaiser]:
        assert cls().dispatch_infer_shape(5) == (5,)
        assert cls().dispatch_infer_shape(DummyItem()) == (5,)

    for cls in [TrilIndices, TrilIndicesFrom, TriuIndices, TriuIndicesFrom]:
        assert cls().dispatch_infer_shape((2, 2)) == (2, 2)
        assert cls().dispatch_infer_shape() == ()

    assert Logspace().infer_shape(1, 10) == (50,)
    assert Logspace().infer_shape(1, 10, num=10) == (10,)
    assert Logspace().infer_shape(1, 10, 10) == (10,)

    assert Frombuffer().infer_shape(count=10) == (10,)
    assert Frombuffer().infer_shape(count=-1) is None
    assert Frombuffer().infer_shape() is None
