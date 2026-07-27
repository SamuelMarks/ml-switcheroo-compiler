# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.space_batch import space_to_batch, space_to_batch_nd


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_space_to_batch_eager(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    assert space_to_batch(t, 2, [[0, 0]]).config.shape == (2, 3)
    assert space_to_batch_nd(t, [2], [[0, 0]]).config.shape == (2, 3)


def test_space_to_batch_tracing(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node", return_value="node")
    mock_op = mocker.MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=mocker.MagicMock(return_value=mock_op))
    assert space_to_batch(t, 2, [[0, 0]]) == "node"
    assert space_to_batch_nd(t, [2], [[0, 0]]) == "node"


def test_space_to_batch_infer_shape():
    from ml_switcheroo_compiler.ops.shape.space_batch import SpaceToBatch, SpaceToBatchND

    assert SpaceToBatch().infer_shape(None, None, None) == ()
    assert SpaceToBatchND().infer_shape(None, None, None) == ()
