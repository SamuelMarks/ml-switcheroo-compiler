"""Tests for test_joining."""

from __future__ import annotations

import typing

import pytest
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.joining import append, column_stack, concatenate, dstack, hstack, stack, vstack
from tests.testing_utils import MockTestStub, ensure_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: BackendArg = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_concatenate(mocker: MockerFixture) -> None:
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
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((6, 3)), raising=False)
    assert ensure_tensor(concatenate([t1, t2], 0)).config.shape == (6, 3)


def test_stack(mocker: MockerFixture) -> None:
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
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((2, 2, 3)), raising=False)
    assert ensure_tensor(stack([t1, t2], 0)).config.shape == (2, 2, 3)


def test_vstack(mocker: MockerFixture) -> None:
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
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((6, 3)), raising=False)
    assert ensure_tensor(vstack([t1, t2])).config.shape == (6, 3)


def test_hstack(mocker: MockerFixture) -> None:
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
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((2, 7)), raising=False)
    assert ensure_tensor(hstack([t1, t2])).config.shape == (2, 7)


def test_dstack(mocker: MockerFixture) -> None:
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
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3, 2)), raising=False)
    assert ensure_tensor(dstack([t1, t2])).config.shape == (2, 3, 2)


def test_append(mocker: MockerFixture) -> None:
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
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((6, 3)), raising=False)
    assert ensure_tensor(append(t1, t2, axis=0)).config.shape == (6, 3)


def test_column_stack(mocker: MockerFixture) -> None:
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


def test_stack_dtype_branches(mocker: MockerFixture) -> None:
    """Test function test_stack_dtype_branches.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.ops.shape.joining._emit_shape_node", return_value="stack")
    config.eager_mode = False

    class FakeDTypeName(MockTestStub):
        """Test class FakeDTypeName."""

        name = "int32"

    t1 = Tensor(
        MockTensor((2, 3)).data,
        TensorConfig((2, 3), FakeDTypeName(), "cpu"),
    )
    stack([t1, t1], 0)

    class FakeDTypeDunderName:
        """Test class FakeDTypeDunderName."""

        __name__ = "int32"

    t2 = Tensor(
        MockTensor((2, 3)).data,
        TensorConfig((2, 3), FakeDTypeDunderName(), "cpu"),
    )
    stack([t2, t2], 0)

    class FakeDTypeValue(MockTestStub):
        """Test class FakeDTypeValue."""

        value = "int32"

    t3 = Tensor(
        MockTensor((2, 3)).data,
        TensorConfig((2, 3), FakeDTypeValue(), "cpu"),
    )
    stack([t3, t3], 0)

    class FakeDTypeInvalid(MockTestStub):
        """Test class FakeDTypeInvalid."""

        pass

    t4 = Tensor(
        MockTensor((2, 3)).data,
        TensorConfig((2, 3), FakeDTypeInvalid(), "cpu"),
    )
    res4 = stack([t4, t4], 0)
    assert str(res4).startswith("stack")


def test_stack_and_concatenate_exact_shape_deduction(mocker: MockerFixture) -> None:
    """Verify that stack and concatenate compute exact output shapes in graph mode.

    Args:
        mocker (MockerFixture): Pytest mocker fixture.
    """
    captured_shapes: dict[str, tuple[int, ...]] = {}

    def mock_emit(
        op_name: str, inputs: list[Tensor], attrs: dict[str, BackendArg], shape: tuple[int, ...], dtype: BackendArg
    ) -> str:
        """Test function mock_emit.

        Args:
            op_name: Test parameter `op_name`.
            inputs: Test parameter `inputs`.
            attrs: Test parameter `attrs`.
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.

        Returns:
            str: Computed output value from `mock_emit`.
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
