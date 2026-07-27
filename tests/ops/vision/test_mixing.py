# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.mixing import cutmix, mixup


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_vision_mixing_funcs(mocker):
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.mixing._emit_shape_node", return_value="node")
    assert mixup(t1, t2) == "node"
    assert cutmix(t1, t2) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.mixing.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))
    assert mixup(t1, t2).config.shape == (2, 3)
    assert cutmix(t1, t2).config.shape == (2, 3)
