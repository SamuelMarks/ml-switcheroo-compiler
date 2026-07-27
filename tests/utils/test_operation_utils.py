"""Tests for operation utilities."""

from unittest.mock import MagicMock

import pytest

from ml_switcheroo_compiler.utils.operation_utils import (
    ExpandDimsInference,
    MeanInference,
    ReshapeInference,
    ShapeInferenceStrategy,
    SplitInference,
    SqueezeInference,
    TransposeInference,
    _validate_squeeze_dims,
    compute_shape_propagation,
    get_source_inputs,
)


def test_get_source_inputs() -> None:
    """Test coverage."""
    # 1. No _keras_history
    tensor = MagicMock(spec=[])
    assert get_source_inputs(tensor) == [tensor]

    # 2. _keras_history but no inputs
    tensor2 = MagicMock()
    tensor2._keras_history.node.operation.inputs = []
    assert get_source_inputs(tensor2) == [tensor2]

    # 3. _keras_history with inputs
    tensor3 = MagicMock()
    inp1 = MagicMock(spec=[])
    tensor3._keras_history.node.operation.inputs = [inp1]
    assert get_source_inputs(tensor3) == [inp1]


def test__validate_squeeze_dims() -> None:
    """Test coverage."""
    with pytest.raises(ValueError, match="is out of bounds"):
        _validate_squeeze_dims((1, 1), (2,))
    with pytest.raises(ValueError, match="is out of bounds"):
        _validate_squeeze_dims((1, 1), (-1,))
    with pytest.raises(ValueError, match="Cannot squeeze dimension"):
        _validate_squeeze_dims((2, 1), (0,))


def test_shape_inference_strategy_base() -> None:
    """Test coverage."""

    class DummyStrategy(ShapeInferenceStrategy):
        """Dummy strategy."""

        def __call__(self, *args, **kwargs):
            """Test function."""
            return super().__call__(*args, **kwargs)

    strategy = DummyStrategy()
    assert strategy((1,), (), {}) is None


def test_reshape_inference() -> None:
    """Test coverage."""
    strategy = ReshapeInference()
    result = strategy((2, 3), (None, (3, 2)), {})
    assert result == (3, 2)


def test_transpose_inference() -> None:
    """Test coverage."""
    strategy = TransposeInference()
    assert strategy((2, 3, 4), (None,), {"axes": (2, 0, 1)}) == (4, 2, 3)
    assert strategy((2, 3, 4), (None,), {}) == (4, 3, 2)
    assert strategy((2, 3, 4), (None, (2, 0, 1)), {}) == (4, 2, 3)


def test_expand_dims_inference() -> None:
    """Test coverage."""
    strategy = ExpandDimsInference()
    assert strategy((2, 3), (None,), {"axis": 1}) == (2, 1, 3)
    assert strategy((2, 3), (None, 1), {}) == (2, 1, 3)
    assert strategy((2, 3), (None,), {"axis": -1}) == (2, 3, 1)


def test_squeeze_inference() -> None:
    """Test coverage."""
    strategy = SqueezeInference()
    assert strategy((1, 2, 1, 3), (None,), {}) == (2, 3)
    assert strategy((1, 2, 1, 3), (None,), {"axis": 0}) == (2, 1, 3)
    assert strategy((1, 2, 1, 3), (None, 0), {}) == (2, 1, 3)
    assert strategy((1, 2, 1, 3), (None, (0, 2)), {}) == (2, 3)


def test_split_inference() -> None:
    """Test coverage."""
    strategy = SplitInference()
    assert strategy((4, 3), (None, 2), {}) == [(2, 3), (2, 3)]
    assert strategy((4, 6), (None, 2), {"axis": 1}) == [(4, 3), (4, 3)]
    assert strategy((4, 6), (None, 2, 1), {}) == [(4, 3), (4, 3)]
    assert strategy((4, 6), (None, (2, 4)), {}) == (4, 6)


def test_mean_inference_helpers() -> None:
    """Test coverage."""
    strategy = MeanInference()
    assert strategy._resolve_axis(None, 3) == set()
    assert strategy._resolve_axis(1, 3) == {1}
    assert strategy._resolve_axis(-1, 3) == {2}
    assert strategy._resolve_axis((0, 2), 3) == {0, 2}
    strategy._validate_datatype_promotion({})
    strategy._validate_datatype_promotion({"dtype": "float32"})
    strategy._validate_datatype_promotion({"dtype": 32})


def test_mean_inference() -> None:
    """Test coverage."""
    strategy = MeanInference()
    assert strategy((2, 3, 4), (None,), {"axis": 1}) == (2, 4)
    assert strategy((2, 3, 4), (None, 1), {}) == (2, 4)
    assert strategy((2, 3, 4), (None,), {"axis": -1}) == (2, 3)
    assert strategy((2, 3, 4), (None, (0, 2)), {}) == (3,)
    assert strategy((2, 3, 4), (None, 1), {"keepdims": True}) == (2, 1, 4)
    assert strategy((2, 3, 4), (None, (0, 2)), {"keepdims": True}) == (1, 3, 1)
    assert strategy((2, 3, 4), (None,), {}) == (2, 3, 4)


def test_compute_shape_propagation() -> None:
    """Test coverage."""
    assert compute_shape_propagation("unknown", (2, 3), (), {}) == (2, 3)
    assert compute_shape_propagation("reshape", (2, 3), (None, (6,)), {}) == (6,)
