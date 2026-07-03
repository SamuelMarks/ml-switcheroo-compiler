"""Tests for operation utilities."""

import pytest

from ml_switcheroo_compiler.utils.operation_utils import (
    ExpandDimsInference,
    MeanInference,
    ReshapeInference,
    ShapeInferenceStrategy,
    SplitInference,
    SqueezeInference,
    TransposeInference,
    compute_shape_propagation,
)


def test_shape_inference_strategy_base() -> object:
    """Test base class raises NotImplementedError."""
    strategy = ShapeInferenceStrategy()
    with pytest.raises(NotImplementedError):
        strategy((1,), (), {})


def test_reshape_inference() -> object:
    """Test reshape inference."""
    strategy = ReshapeInference()
    result = strategy((2, 3), (None, (3, 2)), {})
    assert result == (3, 2)


def test_transpose_inference() -> object:
    """Test transpose inference."""
    strategy = TransposeInference()
    # With axes
    assert strategy((2, 3, 4), (None,), {"axes": (2, 0, 1)}) == (4, 2, 3)
    # Without axes (default reverse)
    assert strategy((2, 3, 4), (None,), {}) == (4, 3, 2)
    # With axes in args
    assert strategy((2, 3, 4), (None, (2, 0, 1)), {}) == (4, 2, 3)


def test_expand_dims_inference() -> object:
    """Test expand_dims inference."""
    strategy = ExpandDimsInference()
    # Axis as kwarg
    assert strategy((2, 3), (None,), {"axis": 1}) == (2, 1, 3)
    # Axis as arg
    assert strategy((2, 3), (None, 1), {}) == (2, 1, 3)
    # Negative axis
    assert strategy((2, 3), (None,), {"axis": -1}) == (2, 3, 1)


def test_squeeze_inference() -> object:
    """Test squeeze inference."""
    strategy = SqueezeInference()
    # No axis
    assert strategy((1, 2, 1, 3), (None,), {}) == (2, 3)
    # Axis as kwarg
    assert strategy((1, 2, 1, 3), (None,), {"axis": 0}) == (2, 1, 3)
    # Axis as arg
    assert strategy((1, 2, 1, 3), (None, 0), {}) == (2, 1, 3)
    # Axis as tuple/list
    assert strategy((1, 2, 1, 3), (None, (0, 2)), {}) == (2, 3)


def test_split_inference() -> object:
    """Test split inference."""
    strategy = SplitInference()
    # Splits, default axis 0
    assert strategy((4, 3), (None, 2), {}) == [(2, 3), (2, 3)]
    # Splits with axis
    assert strategy((4, 6), (None, 2), {"axis": 1}) == [(4, 3), (4, 3)]
    # Splits with axis as arg
    assert strategy((4, 6), (None, 2, 1), {}) == [(4, 3), (4, 3)]
    # Non-int splits
    assert strategy((4, 6), (None, (2, 4)), {}) == (4, 6)


def test_mean_inference_helpers() -> object:
    """Test MeanInference helper methods."""
    strategy = MeanInference()

    # Test _resolve_axis
    assert strategy._resolve_axis(None, 3) == set()
    assert strategy._resolve_axis(1, 3) == {1}
    assert strategy._resolve_axis(-1, 3) == {2}
    assert strategy._resolve_axis((0, 2), 3) == {0, 2}

    # Test _validate_datatype_promotion
    strategy._validate_datatype_promotion({})
    strategy._validate_datatype_promotion({"dtype": "float32"})
    strategy._validate_datatype_promotion({"dtype": 32})  # Should not raise based on current impl


def test_mean_inference() -> object:
    """Test mean inference."""
    strategy = MeanInference()
    # Axis as kwarg
    assert strategy((2, 3, 4), (None,), {"axis": 1}) == (2, 4)
    # Axis as arg
    assert strategy((2, 3, 4), (None, 1), {}) == (2, 4)
    # Negative axis
    assert strategy((2, 3, 4), (None,), {"axis": -1}) == (2, 3)
    # Axis as tuple
    assert strategy((2, 3, 4), (None, (0, 2)), {}) == (3,)
    # Keepdims
    assert strategy((2, 3, 4), (None, 1), {"keepdims": True}) == (2, 1, 4)
    assert strategy((2, 3, 4), (None, (0, 2)), {"keepdims": True}) == (1, 3, 1)
    # No axis fallback
    assert strategy((2, 3, 4), (None,), {}) == (2, 3, 4)


def test_compute_shape_propagation() -> object:
    """Test compute_shape_propagation function dispatcher."""
    # Unknown op
    assert compute_shape_propagation("unknown", (2, 3), (), {}) == (2, 3)
    # Reshape op
    assert compute_shape_propagation("reshape", (2, 3), (None, (6,)), {}) == (6,)
