# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.transforms import elastic_transform, flip_left_right, flip_up_down, perspective_transform


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_vision_transforms_funcs(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.transforms._emit_shape_node", return_value="node")
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.vision.transforms.get_op")
    mock_op = mocker.MagicMock()
    mock_get_op.return_value = mocker.MagicMock(return_value=mock_op)
    assert perspective_transform(t, t, t) == "node"
    assert elastic_transform(t, t) == "node"
    assert flip_left_right(t) == mock_op()
    assert flip_up_down(t) == mock_op()
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.transforms.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))
    assert perspective_transform(t, t, t).config.shape == (2, 3)
    assert elastic_transform(t, t).config.shape == (2, 3)
    assert flip_left_right(t).config.shape == (2, 3)
    assert flip_up_down(t).config.shape == (2, 3)


def test_vision_transforms_funcs_kwargs(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.transforms._emit_shape_node", return_value="node")
    assert perspective_transform(t, t, t, config_obj="cfg") == "node"
    assert elastic_transform(t, t, config_obj="cfg") == "node"


def test_transforms_opdef_infer_shape():
    from ml_switcheroo_compiler.ops.vision.transforms import ElasticTransform

    class DummyTensor:
        shape = (1, 2, 3)

    t = DummyTensor()
    assert ElasticTransform().infer_shape(t) == (1, 2, 3)
