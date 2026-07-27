# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.interpolation import map_coordinates, resize, resize_bicubic, resize_bilinear, resize_lanczos3, resize_nearest


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_vision_interpolation_funcs(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.interpolation._emit_shape_node", return_value="node")
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.registry.get_op")
    mock_op = mocker.MagicMock()
    mock_get_op.return_value = mocker.MagicMock(return_value=mock_op)
    assert resize_bilinear(t, (10, 10)) == "node"
    assert resize_nearest(t, (10, 10)) == "node"
    assert resize_bicubic(t, (10, 10)) == "node"
    assert resize_lanczos3(t, (10, 10)) == "node"
    assert resize(t, (10, 10)) == "node"
    assert map_coordinates(t, t, 1) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.interpolation.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))
    assert resize_bilinear(t, (10, 10)).config.shape == (2, 3)
    assert resize_nearest(t, (10, 10)).config.shape == (2, 3)
    assert resize_bicubic(t, (10, 10)).config.shape == (2, 3)
    assert resize_lanczos3(t, (10, 10)).config.shape == (2, 3)
    assert resize(t, (10, 10)).config.shape == (2, 3)
    assert map_coordinates(t, t, 1).config.shape == (2, 3)
