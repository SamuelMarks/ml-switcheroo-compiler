"""Tests for test_splitting."""

from __future__ import annotations

import typing

# ruff: noqa: E501
import pytest
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.splitting import (
    GetItemOp,
    Unstack,
    _calculate_num_splits,
    _split_even,
    _split_sections,
    _validate_split_axis,
    array_split,
    dsplit,
    hsplit,
    split,
    unstack,
    vsplit,
)
from tests.testing_utils import MockTestStub, coerce_operand_list, ensure_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor(MockTestStub):
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


def test_calculate_num_splits() -> None:
    """Test function test_calculate_num_splits."""
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    assert _calculate_num_splits(t, 2, 0) == 3
    assert _calculate_num_splits(t, [2, 4], 0) == 3
    assert _calculate_num_splits(t, 3, 0) == 2


def test_validate_split_axis() -> None:
    """Test function test_validate_split_axis."""
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    assert _validate_split_axis(t, 1) == 1
    assert _validate_split_axis(t, -1) == -1


def test_split_even(mocker: MockerFixture) -> None:
    """Test function test_split_even.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="split_even")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    pytest.MonkeyPatch().setattr(mock_item, "dtype", "float32", raising=False)
    pytest.MonkeyPatch().setattr(mock_item, "device", "cpu", raising=False)
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(coerce_operand_list(_split_even(t, 2, 0, 3))) == 3
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(
        mock_backend.execute_op,
        "return_value",
        [MockTensor((2, 4)), MockTensor((2, 4)), MockTensor((2, 4))],
        raising=False,
    )
    pytest.MonkeyPatch().setattr(mock_backend.array, "side_effect", lambda x: MockTensor(), raising=False)
    res = _split_even(t, 2, 0, 3)
    assert len(coerce_operand_list(res)) == 3
    assert ensure_tensor(coerce_operand_list(res)[0]).config.shape == (2, 4)


def test_split_sections(mocker: MockerFixture) -> None:
    """Test function test_split_sections.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="split_sec")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    pytest.MonkeyPatch().setattr(mock_item, "dtype", "float32", raising=False)
    pytest.MonkeyPatch().setattr(mock_item, "device", "cpu", raising=False)
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(coerce_operand_list(_split_sections(t, [2, 4], 0, 3))) == 3
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(
        mock_backend.execute_op, "return_value", [MockTensor((2, 4)), MockTensor((4, 4))], raising=False
    )
    pytest.MonkeyPatch().setattr(mock_backend.array, "side_effect", lambda x: MockTensor(), raising=False)
    res = _split_sections(t, [2, 4], 0, 2)
    assert len(coerce_operand_list(res)) == 2
    assert ensure_tensor(coerce_operand_list(res)[0]).config.shape == (2, 4)


def test_split(mocker: MockerFixture) -> None:
    """Test function test_split.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._split_even", return_value="even")
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._split_sections", return_value="sections")
    assert split(t, 2, 0) == "even"
    assert split(t, [2, 4], 0) == "sections"


def test_unstack(mocker: MockerFixture) -> None:
    """Test function test_unstack.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="unstack")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    pytest.MonkeyPatch().setattr(mock_item, "dtype", "float32", raising=False)
    pytest.MonkeyPatch().setattr(mock_item, "device", "cpu", raising=False)
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(coerce_operand_list(unstack(t, 0))) == 6

    # 0-D unstack in tracing mode (hits rank == 0 fallback: n_slices = 1, out_shape = ())
    t_scalar = Tensor(MockTensor(()).data, TensorConfig((), "float32", "cpu"))
    assert len(coerce_operand_list(unstack(t_scalar, 0))) == 1

    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(
        mock_backend.execute_op, "return_value", [MockTensor((4,)) for _ in range(6)], raising=False
    )
    pytest.MonkeyPatch().setattr(mock_backend.array, "side_effect", lambda x: MockTensor(), raising=False)
    res = unstack(t, 0)
    assert len(coerce_operand_list(res)) == 6
    assert ensure_tensor(coerce_operand_list(res)[0]).config.shape == (4,)


def test_unstack_exact_shape_deduction(mocker: MockerFixture) -> None:
    """Verify that unstack deduces exact shapes across different axes in graph mode.

    Args:
        mocker (MockerFixture): Pytest mocker fixture.
    """
    captured: list[tuple[str, tuple[int, ...]]] = []

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
        captured.append((op_name, shape))
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", side_effect=mock_emit)
    config.eager_mode = False

    t = Tensor(None, TensorConfig((3, 5, 7), "float32", "cpu"))

    # Unstack along axis 0: 3 slices of shape (5, 7)
    res0 = unstack(t, axis=0)
    assert len(coerce_operand_list(res0)) == 3
    assert len(captured) == 3
    assert all(shape == (5, 7) for _, shape in captured)

    # Unstack along axis 1: 5 slices of shape (3, 7)
    captured.clear()
    res1 = unstack(t, axis=1)
    assert len(coerce_operand_list(res1)) == 5
    assert len(captured) == 5
    assert all(shape == (3, 7) for _, shape in captured)

    # Unstack along negative axis -1: 7 slices of shape (3, 5)
    captured.clear()
    res2 = unstack(t, axis=-1)
    assert len(coerce_operand_list(res2)) == 7
    assert len(captured) == 7
    assert all(shape == (3, 5) for _, shape in captured)


def test_array_split(mocker: MockerFixture) -> None:
    """Test function test_array_split.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="array_split")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    pytest.MonkeyPatch().setattr(mock_item, "dtype", "float32", raising=False)
    pytest.MonkeyPatch().setattr(mock_item, "device", "cpu", raising=False)
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(coerce_operand_list(array_split(t, 2, 0))) == 1
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(
        mock_backend.execute_op, "return_value", [MockTensor((3, 4)), MockTensor((3, 4))], raising=False
    )
    pytest.MonkeyPatch().setattr(mock_backend.array, "side_effect", lambda x: MockTensor(), raising=False)
    res = array_split(t, 2, 0)
    assert len(coerce_operand_list(res)) == 2
    assert ensure_tensor(coerce_operand_list(res)[0]).config.shape == (3, 4)


def test_classes_infer_shape() -> None:
    """Test function test_classes_infer_shape."""
    op = Unstack()
    assert op.infer_shape(MockTensor((2, 3)), axis=0) == ((3,), (3,))


def test_unstack_infer_shape_extra() -> None:
    """Test function test_unstack_infer_shape_extra."""
    op2 = GetItemOp()
    assert op2.infer_shape("not tensor") == ()

    class NoShape(MockTestStub):
        """Test class NoShape."""

        pass

    assert op2.infer_shape(NoShape()) == ()


def test_vsplit_hsplit_dsplit_scalar(mocker: MockerFixture) -> None:
    """Test function test_vsplit_hsplit_dsplit_scalar.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor(()).data, TensorConfig((), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="split")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(coerce_operand_list(vsplit(t, 2))) == 2
    assert len(coerce_operand_list(hsplit(t, 2))) == 2
    assert len(coerce_operand_list(dsplit(t, 2))) == 2


def test_vsplit_impl(mocker: MockerFixture) -> None:
    """Test function test_vsplit_impl.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="vsplit")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    pytest.MonkeyPatch().setattr(mock_item, "dtype", "float32", raising=False)
    pytest.MonkeyPatch().setattr(mock_item, "device", "cpu", raising=False)
    mock_builder.emit_tracing_node.return_value = mock_item
    res1 = vsplit(t, 2)
    assert len(coerce_operand_list(res1)) == 2
    res2 = vsplit(t, [2, 4])
    assert len(coerce_operand_list(res2)) == 3
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(
        mock_backend.execute_op, "return_value", [MockTensor((3, 4)), MockTensor((3, 4))], raising=False
    )
    assert len(coerce_operand_list(vsplit(t, 2))) == 2


def test_hsplit_impl(mocker: MockerFixture) -> None:
    """Test function test_hsplit_impl.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="hsplit")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    pytest.MonkeyPatch().setattr(mock_item, "dtype", "float32", raising=False)
    pytest.MonkeyPatch().setattr(mock_item, "device", "cpu", raising=False)
    mock_builder.emit_tracing_node.return_value = mock_item
    res1 = hsplit(t, 2)
    assert len(coerce_operand_list(res1)) == 2
    res2 = hsplit(t, [2])
    assert len(coerce_operand_list(res2)) == 2
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(
        mock_backend.execute_op, "return_value", [MockTensor((6, 2)), MockTensor((6, 2))], raising=False
    )
    assert len(coerce_operand_list(hsplit(t, 2))) == 2


def test_dsplit_impl(mocker: MockerFixture) -> None:
    """Test function test_dsplit_impl.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((6, 4, 2)).data, TensorConfig((6, 4, 2), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="dsplit")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    pytest.MonkeyPatch().setattr(mock_item, "dtype", "float32", raising=False)
    pytest.MonkeyPatch().setattr(mock_item, "device", "cpu", raising=False)
    mock_builder.emit_tracing_node.return_value = mock_item
    res1 = dsplit(t, 2)
    assert len(coerce_operand_list(res1)) == 2
    res2 = dsplit(t, [1])
    assert len(coerce_operand_list(res2)) == 2
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(
        mock_backend.execute_op, "return_value", [MockTensor((6, 4, 1)), MockTensor((6, 4, 1))], raising=False
    )
    assert len(coerce_operand_list(dsplit(t, 2))) == 2


def test_calculate_num_splits_extra() -> None:
    """Test function test_calculate_num_splits_extra."""

    class NoShapeTensor(MockTestStub):
        """Test class NoShapeTensor."""

        pass

    assert _calculate_num_splits(NoShapeTensor(), 2, 0) == 2
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    assert _calculate_num_splits(t, 0, 0) == 1
    assert _calculate_num_splits(t, 5, 0) == 5


def test_validate_split_axis_extra() -> None:
    """Test function test_validate_split_axis_extra."""

    class NoShapeTensor(MockTestStub):
        """Test class NoShapeTensor."""

        pass

    assert _validate_split_axis(NoShapeTensor(), 1) == 1
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    with pytest.raises(ValueError):
        _validate_split_axis(t, 2)
