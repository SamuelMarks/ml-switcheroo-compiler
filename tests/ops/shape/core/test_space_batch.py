"""Tests for test_space_batch."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.space_batch import space_to_batch, space_to_batch_nd

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__."""
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_space_to_batch_eager(mocker: EagerTensorProtocol) -> None:
    """Test function test_space_to_batch_eager."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    assert cast("EagerTensorProtocol", getattr(space_to_batch(t, 2, [[0, 0]]), "config")).shape == (2, 3)
    assert cast("EagerTensorProtocol", getattr(space_to_batch_nd(t, [2], [[0, 0]]), "config")).shape == (2, 3)


def test_space_to_batch_tracing(mocker: EagerTensorProtocol) -> None:
    """Test function test_space_to_batch_tracing."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="node")
    mock_op = mocker.MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=mocker.MagicMock(return_value=mock_op))
    assert space_to_batch(t, 2, [[0, 0]]) == "node"
    assert space_to_batch_nd(t, [2], [[0, 0]]) == "node"


def test_space_to_batch_infer_shape() -> None:
    """Test function test_space_to_batch_infer_shape."""
    from ml_switcheroo_compiler.ops.shape.space_batch import SpaceToBatch, SpaceToBatchND

    assert cast("Callable[..., EagerTensorProtocol]", SpaceToBatch)().infer_shape(None, None, None) == ()
    assert cast("Callable[..., EagerTensorProtocol]", SpaceToBatchND)().infer_shape(None, None, None) == ()
