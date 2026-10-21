"""Tests for test_scatter."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.scatter import scatter, scatter_add, scatter_nd, tensor_scatter_add, tensor_scatter_max, tensor_scatter_min, tensor_scatter_update
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_scatter(mocker: EagerTensorProtocol) -> None:
    """Test function test_scatter.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    src = Tensor([1], TensorConfig((1,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.scatter._emit_shape_node", return_value="scatter")
    assert scatter(t, 0, idx, src) == "scatter"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.scatter.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    res = scatter(t, 0, idx, src)
    assert as_tensor(getattr(res, "config")).shape == (2, 3)


def test_scatter_nd(mocker: EagerTensorProtocol) -> None:
    """Test function test_scatter_nd.

    Args:
        mocker: Test parameter `mocker`.
    """
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    src = Tensor([1], TensorConfig((1,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.scatter._emit_shape_node", return_value="scatter_nd")
    assert scatter_nd(idx, src, (2, 3)) == "scatter_nd"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.scatter.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert as_tensor(getattr(scatter_nd(idx, src, (2, 3)), "config")).shape == (2, 3)


def test_scatter_add(mocker: EagerTensorProtocol) -> None:
    """Test function test_scatter_add.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    src = Tensor([1], TensorConfig((1,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.scatter._emit_shape_node", return_value="scatter_add")
    assert scatter_add(t, 0, idx, src) == "scatter_add"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.scatter.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert as_tensor(getattr(scatter_add(t, 0, idx, src), "config")).shape == (2, 3)


def test_tensor_scatter_update(mocker: EagerTensorProtocol) -> None:
    """Test function test_tensor_scatter_update.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    src = Tensor([1], TensorConfig((1,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.scatter._emit_shape_node", return_value="ts_update")
    assert tensor_scatter_update(t, idx, src) == "ts_update"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.scatter.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert as_tensor(getattr(tensor_scatter_update(t, idx, src), "config")).shape == (2, 3)


def test_tensor_scatter_max(mocker: EagerTensorProtocol) -> None:
    """Test function test_tensor_scatter_max.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    src = Tensor([1], TensorConfig((1,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.scatter._emit_shape_node", return_value="ts_max")
    assert tensor_scatter_max(t, idx, src) == "ts_max"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.scatter.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert as_tensor(getattr(tensor_scatter_max(t, idx, src), "config")).shape == (2, 3)


def test_tensor_scatter_min(mocker: EagerTensorProtocol) -> None:
    """Test function test_tensor_scatter_min.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    src = Tensor([1], TensorConfig((1,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.scatter._emit_shape_node", return_value="ts_min")
    assert tensor_scatter_min(t, idx, src) == "ts_min"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.scatter.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert as_tensor(getattr(tensor_scatter_min(t, idx, src), "config")).shape == (2, 3)


def test_tensor_scatter_add(mocker: EagerTensorProtocol) -> None:
    """Test function test_tensor_scatter_add.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    idx = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    src = Tensor([1], TensorConfig((1,), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.scatter._emit_shape_node", return_value="ts_add")
    assert tensor_scatter_add(t, idx, src) == "ts_add"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.scatter.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", "res")
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert as_tensor(getattr(tensor_scatter_add(t, idx, src), "config")).shape == (2, 3)


def test_scatter_exact_shape_deduction(mocker: EagerTensorProtocol) -> None:
    """Verify that scatter and tensor_scatter operations deduce exact shapes in graph mode.

    Args:
        mocker (EagerTensorProtocol): Pytest mocker fixture.
    """
    captured_shapes: dict[str, tuple[int, ...]] = {}

    def mock_emit(op_name: str, inputs: list[Tensor], attrs: dict[str, EagerValue], shape: tuple[int, ...], dtype: EagerValue) -> str:
        """Test function mock_emit.

        Args:
            op_name: Test parameter `op_name`.
            inputs: Test parameter `inputs`.
            attrs: Test parameter `attrs`.
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.
        """
        captured_shapes[op_name] = shape
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.shape.scatter._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False

    t = Tensor(None, TensorConfig((4, 5, 6), "float32", "cpu"))
    idx = Tensor(None, TensorConfig((2, 5, 6), "int32", "cpu"))
    src = Tensor(None, TensorConfig((2, 5, 6), "float32", "cpu"))

    scatter(t, axis=0, index=idx, src=src)
    assert captured_shapes["Scatter"] == (4, 5, 6)

    scatter_add(t, axis=0, index=idx, src=src)
    assert captured_shapes["ScatterAdd"] == (4, 5, 6)

    tensor_scatter_update(t, idx, src)
    assert captured_shapes["TensorScatterUpdate"] == (4, 5, 6)

    tensor_scatter_max(t, idx, src)
    assert captured_shapes["TensorScatterMax"] == (4, 5, 6)

    tensor_scatter_min(t, idx, src)
    assert captured_shapes["TensorScatterMin"] == (4, 5, 6)

    tensor_scatter_add(t, idx, src)
    assert captured_shapes["TensorScatterAdd"] == (4, 5, 6)
