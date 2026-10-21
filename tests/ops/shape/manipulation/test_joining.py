"""Tests for test_joining."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.joining import append, column_stack, concatenate, dstack, hstack, stack, vstack
from tests.testing_utils import as_callable, as_tensor

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


def test_concatenate(mocker: EagerTensorProtocol) -> None:
    """Test function test_concatenate.

    Args:
        mocker: Test parameter `mocker`.
    """
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((4, 3)).data, TensorConfig((4, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="concat")
    assert concatenate([t1, t2], 0) == "concat"

    # 0-D scalar tensor concatenation in tracing mode
    t_scalar = Tensor(MockTensor(()).data, TensorConfig((), "float32", "cpu"))
    assert concatenate([t_scalar, t_scalar], 0) == "concat"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((6, 3)))
    assert as_tensor(getattr(concatenate([t1, t2], 0), "config")).shape == (6, 3)


def test_stack(mocker: EagerTensorProtocol) -> None:
    """Test function test_stack.

    Args:
        mocker: Test parameter `mocker`.
    """
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="stack")
    assert stack([t1, t2], 0) == "stack"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 2, 3)))
    assert as_tensor(getattr(stack([t1, t2], 0), "config")).shape == (2, 2, 3)


def test_vstack(mocker: EagerTensorProtocol) -> None:
    """Test function test_vstack.

    Args:
        mocker: Test parameter `mocker`.
    """
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((4, 3)).data, TensorConfig((4, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="vstack")
    assert vstack([t1, t2]) == "vstack"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((6, 3)))
    assert as_tensor(getattr(vstack([t1, t2]), "config")).shape == (6, 3)


def test_hstack(mocker: EagerTensorProtocol) -> None:
    """Test function test_hstack.

    Args:
        mocker: Test parameter `mocker`.
    """
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((2, 4)).data, TensorConfig((2, 4), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="hstack")
    assert hstack([t1, t2]) == "hstack"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 7)))
    assert as_tensor(getattr(hstack([t1, t2]), "config")).shape == (2, 7)


def test_dstack(mocker: EagerTensorProtocol) -> None:
    """Test function test_dstack.

    Args:
        mocker: Test parameter `mocker`.
    """
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="dstack")
    assert dstack([t1, t2]) == "dstack"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3, 2)))
    assert as_tensor(getattr(dstack([t1, t2]), "config")).shape == (2, 3, 2)


def test_append(mocker: EagerTensorProtocol) -> None:
    """Test function test_append.

    Args:
        mocker: Test parameter `mocker`.
    """
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((4, 3)).data, TensorConfig((4, 3), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining.concatenate", return_value="concat")
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining.reshape", side_effect=lambda x, shape: x)
    assert append(t1, t2, axis=0) == "concat"
    assert append(t1, t2, axis=None) == "concat"

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.joining.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((6, 3)))
    assert as_tensor(getattr(append(t1, t2, axis=0), "config")).shape == (6, 3)


def test_column_stack(mocker: EagerTensorProtocol) -> None:
    """Test function test_column_stack.

    Args:
        mocker: Test parameter `mocker`.
    """
    t1 = Tensor(MockTensor((3,)).data, TensorConfig((3,), "float32", "cpu"))
    t2 = Tensor(MockTensor((3, 2)).data, TensorConfig((3, 2), "float32", "cpu"))

    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining.hstack", return_value="hstack")
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining.reshape", side_effect=lambda x, shape: x)
    assert column_stack([t1, t2]) == "hstack"


def test_stack_dtype_branches(mocker: EagerTensorProtocol) -> None:
    """Test function test_stack_dtype_branches.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="stack")
    config.eager_mode = False

    class FakeDTypeName:
        """Test class FakeDTypeName."""

        name = "int32"

    t1 = as_callable(Tensor)(as_callable(MockTensor)((2, 3)).data, as_callable(TensorConfig)((2, 3), as_callable(FakeDTypeName)(), "cpu"))
    as_callable(stack)([t1, t1], 0)

    class FakeDTypeDunderName:
        """Test class FakeDTypeDunderName."""

        __name__ = "int32"

    t2 = as_callable(Tensor)(as_callable(MockTensor)((2, 3)).data, as_callable(TensorConfig)((2, 3), as_callable(FakeDTypeDunderName)(), "cpu"))
    as_callable(stack)([t2, t2], 0)

    class FakeDTypeValue:
        """Test class FakeDTypeValue."""

        value = "int32"

    t3 = as_callable(Tensor)(as_callable(MockTensor)((2, 3)).data, as_callable(TensorConfig)((2, 3), as_callable(FakeDTypeValue)(), "cpu"))
    as_callable(stack)([t3, t3], 0)

    class FakeDTypeInvalid:
        """Test class FakeDTypeInvalid."""

        pass

    t4 = as_callable(Tensor)(as_callable(MockTensor)((2, 3)).data, as_callable(TensorConfig)((2, 3), as_callable(FakeDTypeInvalid)(), "cpu"))
    res4 = as_callable(stack)([t4, t4], 0)
    assert res4 is not None


def test_stack_and_concatenate_exact_shape_deduction(mocker: EagerTensorProtocol) -> None:
    """Verify that stack and concatenate compute exact output shapes in graph mode.

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

    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False

    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t3 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))

    # Test stack along axis 0
    stack([t1, t2, t3], axis=0)
    assert captured_shapes["Stack"] == (3, 2, 3)

    # Test stack along axis 1
    stack([t1, t2, t3], axis=1)
    assert captured_shapes["Stack"] == (2, 3, 3)

    # Test stack along negative axis -1
    stack([t1, t2, t3], axis=-1)
    assert captured_shapes["Stack"] == (2, 3, 3)

    # Test concatenate along axis 0
    t_cat1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t_cat2 = Tensor(MockTensor((4, 3)).data, TensorConfig((4, 3), "float32", "cpu"))
    concatenate([t_cat1, t_cat2], axis=0)
    assert captured_shapes["Concatenate"] == (6, 3)

    # Test concatenate along axis 1
    t_cat3 = Tensor(MockTensor((2, 5)).data, TensorConfig((2, 5), "float32", "cpu"))
    concatenate([t_cat1, t_cat3], axis=1)
    assert captured_shapes["Concatenate"] == (2, 8)
