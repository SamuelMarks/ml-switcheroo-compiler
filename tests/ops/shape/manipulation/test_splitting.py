# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.splitting import GetItemOp, Unstack, _calculate_num_splits, _split_even, _split_sections, _validate_split_axis, array_split, dsplit, hsplit, split, unstack, vsplit


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_calculate_num_splits():
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    assert _calculate_num_splits(t, 2, 0) == 3
    assert _calculate_num_splits(t, [2, 4], 0) == 3
    assert _calculate_num_splits(t, 3, 0) == 2


def test_validate_split_axis():
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    assert _validate_split_axis(t, 1) == 1
    assert _validate_split_axis(t, -1) == -1


def test_split_even(mocker):
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="split_even")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    mock_item.dtype = "float32"
    mock_item.device = "cpu"
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(_split_even(t, 2, 0, 3)) == 3
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    mock_backend.execute_op.return_value = [MockTensor((2, 4)), MockTensor((2, 4)), MockTensor((2, 4))]
    mock_backend.array.side_effect = lambda x: MockTensor()
    res = _split_even(t, 2, 0, 3)
    assert len(res) == 3
    assert res[0].config.shape == (2, 4)


def test_split_sections(mocker):
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="split_sec")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    mock_item.dtype = "float32"
    mock_item.device = "cpu"
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(_split_sections(t, [2, 4], 0, 3)) == 3
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    mock_backend.execute_op.return_value = [MockTensor((2, 4)), MockTensor((4, 4))]
    mock_backend.array.side_effect = lambda x: MockTensor()
    res = _split_sections(t, [2, 4], 0, 2)
    assert len(res) == 2
    assert res[0].config.shape == (2, 4)


def test_split(mocker):
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._split_even", return_value="even")
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._split_sections", return_value="sections")
    assert split(t, 2, 0) == "even"
    assert split(t, [2, 4], 0) == "sections"


def test_unstack(mocker):
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="unstack")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    mock_item.dtype = "float32"
    mock_item.device = "cpu"
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(unstack(t, 0)) == 1
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    mock_backend.execute_op.return_value = [MockTensor((4,)) for _ in range(6)]
    mock_backend.array.side_effect = lambda x: MockTensor()
    res = unstack(t, 0)
    assert len(res) == 6
    assert res[0].config.shape == (4,)


def test_array_split(mocker):
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="array_split")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    mock_item.dtype = "float32"
    mock_item.device = "cpu"
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(array_split(t, 2, 0)) == 1
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    mock_backend.execute_op.return_value = [MockTensor((3, 4)), MockTensor((3, 4))]
    mock_backend.array.side_effect = lambda x: MockTensor()
    res = array_split(t, 2, 0)
    assert len(res) == 2
    assert res[0].config.shape == (3, 4)


def test_classes_infer_shape():
    op = Unstack()
    assert op.infer_shape(MockTensor((2, 3)), axis=0) == ((3,), (3,))


def test_unstack_infer_shape_extra():
    op2 = GetItemOp()
    assert op2.infer_shape("not tensor") == ()

    class NoShape:
        pass

    assert op2.infer_shape(NoShape()) == ()


def test_vsplit_hsplit_dsplit_scalar(mocker):
    t = Tensor(MockTensor(()).data, TensorConfig((), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="split")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    mock_builder.emit_tracing_node.return_value = mock_item
    assert len(vsplit(t, 2)) == 2
    assert len(hsplit(t, 2)) == 2
    assert len(dsplit(t, 2)) == 2


def test_vsplit_impl(mocker):
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="vsplit")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    mock_item.dtype = "float32"
    mock_item.device = "cpu"
    mock_builder.emit_tracing_node.return_value = mock_item
    res1 = vsplit(t, 2)
    assert len(res1) == 2
    res2 = vsplit(t, [2, 4])
    assert len(res2) == 3
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    mock_backend.execute_op.return_value = [MockTensor((3, 4)), MockTensor((3, 4))]
    assert len(vsplit(t, 2)) == 2


def test_hsplit_impl(mocker):
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="hsplit")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    mock_item.dtype = "float32"
    mock_item.device = "cpu"
    mock_builder.emit_tracing_node.return_value = mock_item
    res1 = hsplit(t, 2)
    assert len(res1) == 2
    res2 = hsplit(t, [2])
    assert len(res2) == 2
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    mock_backend.execute_op.return_value = [MockTensor((6, 2)), MockTensor((6, 2))]
    assert len(hsplit(t, 2)) == 2


def test_dsplit_impl(mocker):
    t = Tensor(MockTensor((6, 4, 2)).data, TensorConfig((6, 4, 2), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.splitting._emit_shape_node", return_value="dsplit")
    mock_builder = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.builder.TracingNodeBuilder")
    mock_item = mocker.MagicMock()
    mock_item.dtype = "float32"
    mock_item.device = "cpu"
    mock_builder.emit_tracing_node.return_value = mock_item
    res1 = dsplit(t, 2)
    assert len(res1) == 2
    res2 = dsplit(t, [1])
    assert len(res2) == 2
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.shape.splitting.get_active_backend").return_value
    mock_backend.execute_op.return_value = [MockTensor((6, 4, 1)), MockTensor((6, 4, 1))]
    assert len(dsplit(t, 2)) == 2


def test_calculate_num_splits_extra():

    class NoShapeTensor:
        pass

    assert _calculate_num_splits(NoShapeTensor(), 2, 0) == 2
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    assert _calculate_num_splits(t, 0, 0) == 1
    assert _calculate_num_splits(t, 5, 0) == 5


def test_validate_split_axis_extra():

    class NoShapeTensor:
        pass

    assert _validate_split_axis(NoShapeTensor(), 1) == 1
    t = Tensor(MockTensor((6, 4)).data, TensorConfig((6, 4), "float32", "cpu"))
    with pytest.raises(ValueError):
        _validate_split_axis(t, 2)
