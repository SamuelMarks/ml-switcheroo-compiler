# ruff: noqa: E501
"""Tests for descriptive statistics operations and shape inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.stats.descriptive import (
    ApplyOverAxes,
    Average,
    Bincount,
    ConfusionMatrix,
    Corrcoef,
    Correlate,
    Cov,
    Descriptive,
    Distributions,
    Mean,
    Std,
    TrapezoidalIntegral,
    Variance,
    confusion_matrix,
    descriptive,
    distributions,
    moments,
    trapezoidal_integral,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Mock tensor object with shape property."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__."""
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


class MockShapeMeta:
    """Mock tensor object with shape_metadata property."""

    def __init__(self, shape_metadata: EagerValue = ()) -> None:
        """Test function __init__."""
        self.shape_metadata = shape_metadata


class PlainObj:
    """Plain object without shape information."""

    pass


def test_classes_infer_shape() -> None:
    """Verify infer_shape across all statistical operation classes and branches."""
    # ApplyOverAxes
    aoa = ApplyOverAxes()
    assert aoa.infer_shape() == ()
    assert cast("Callable[..., EagerTensorProtocol]", aoa.infer_shape)(func=lambda x: x, a=cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3, 4)), axes=1) == (2, 1, 4)
    assert cast("Callable[..., EagerTensorProtocol]", aoa.infer_shape)(func=cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3, 4)), axes=-1) == (2, 3, 1)
    assert cast("Callable[..., EagerTensorProtocol]", aoa.infer_shape)(x=cast("Callable[..., EagerTensorProtocol]", MockTensor)((2, 3)), axes=0) == (1, 3)
    assert aoa.infer_shape(a=(2, 3, 4), axes=[0, 2]) == (1, 3, 1)
    assert aoa.infer_shape(a=[2, 3, 4], axes=(1,)) == (2, 1, 4)
    assert cast("Callable[..., EagerTensorProtocol]", aoa.infer_shape)(a=cast("Callable[..., EagerTensorProtocol]", MockShapeMeta)((5, 6)), axes=0) == (1, 6)
    assert cast("Callable[..., EagerTensorProtocol]", aoa.infer_shape)(a=cast("Callable[..., EagerTensorProtocol]", MockShapeMeta)(()), axes=0) == ()
    assert cast("Callable[..., EagerTensorProtocol]", aoa.infer_shape)(a=cast("Callable[..., EagerTensorProtocol]", PlainObj)(), axes=0) == ()
    assert aoa.infer_shape(a=(2, 3), axes=99) == (2, 3)
    assert aoa.infer_shape(a=(2, 3), axes=None) == (2, 3)
    assert aoa.infer_shape(a=(), axes=0) == ()

    # Bincount, Corrcoef, Correlate, Cov
    assert Bincount().infer_shape() == (None,)
    assert Corrcoef().infer_shape() == (None, None)
    assert Correlate().infer_shape() == (None,)
    assert Cov().infer_shape() == (None, None)

    # TrapezoidalIntegral
    ti = TrapezoidalIntegral()
    assert ti.infer_shape((2, 3), axis=0) == (3,)
    assert ti.infer_shape((2, 3), axis=-1) == (2,)
    assert ti.infer_shape(y=(4, 5, 6), axis=1) == (4, 6)

    # ConfusionMatrix
    cm = ConfusionMatrix()
    assert cm.infer_shape(num_classes=5) == (5, 5)
    assert cm.infer_shape() == (0, 0)

    # Base classes check
    assert Mean.op_name == "Mean"
    assert Average.op_name == "Average"
    assert Variance.op_name == "Variance"
    assert Std.op_name == "Std"
    assert Descriptive.op_name == "Descriptive"
    assert Distributions.op_name == "Distributions"


def test_funcs(mocker: EagerTensorProtocol) -> None:
    """Test eager dispatcher wrappers and moments utility function."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    prev_mode = config.eager_mode
    try:
        config.eager_mode = False
        mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.stats.descriptive.get_op")
        mock_op = mocker.MagicMock()
        setattr(mock_get_op, "return_value", mocker.MagicMock(return_value=mock_op))

        assert trapezoidal_integral(t, x=t, dx=2.0, axis=1) == mock_op()
        assert confusion_matrix(t, t, 5, weights=t) == mock_op()
        (m1, v1) = cast("Callable[..., EagerTensorProtocol]", moments)(t, axes=[0], keepdims=True)
        assert m1 == mock_op()
        assert v1 == mock_op()
        assert descriptive(t) == mock_op()
        assert distributions(t) == mock_op()
    finally:
        config.eager_mode = prev_mode
