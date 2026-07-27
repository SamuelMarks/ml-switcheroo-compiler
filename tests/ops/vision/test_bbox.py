# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.bbox import _extract_bounding_boxes_eager, crop, crop_and_resize, crop_images, draw_bounding_boxes, extract_bounding_boxes, extract_patches, pad_images, pad_to_bounding_box


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_vision_bbox_funcs(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.bbox._emit_shape_node", return_value="node")
    assert crop_and_resize(t, t, t, (10, 10)) == "node"
    assert extract_bounding_boxes(t, t, t, crop_size=10) == "node"
    assert crop(t, 1, 1, 10, 10) == "node"
    assert pad_to_bounding_box(t, 1, 1, 10, 10) == "node"
    assert draw_bounding_boxes(t, t, t) == "node"
    assert crop_images(t, (1, 1, 1, 1)) == "node"
    assert extract_patches(t, (10, 10)) == "node"
    assert pad_images(t, (1, 1, 1, 1), (10, 10)) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))
    assert crop_and_resize(t, t, t, (10, 10)).config.shape == (2, 3)
    assert extract_bounding_boxes(t, t, t, crop_size=10).config.shape == (2, 3)
    assert crop(t, 1, 1, 10, 10).config.shape == (2, 3)
    assert pad_to_bounding_box(t, 1, 1, 10, 10).config.shape == (2, 3)
    assert draw_bounding_boxes(t, t, t).config.shape == (2, 3)
    assert crop_images(t, (1, 1, 1, 1)).config.shape == (2, 3)
    assert extract_patches(t, (10, 10)).config.shape == (2, 3)
    assert pad_images(t, (1, 1, 1, 1), (10, 10)).config.shape == (2, 3)


def test_extract_bounding_boxes_eager_extra(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    mock_backend.execute_op.return_value = MockTensor((2, 3))
    mock_backend.array.side_effect = lambda x: MockTensor((2, 3))
    _extract_bounding_boxes_eager(t, t, t, None)


def test_extract_bounding_boxes_kwargs(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.bbox._emit_shape_node", return_value="node")
    assert extract_bounding_boxes(t, t, t, config_obj="cfg") == "node"
    assert extract_bounding_boxes(t, t, t, crop_size=(10, 10)) == "node"


def test_bbox_opdef_infer_shape():
    from ml_switcheroo_compiler.ops.vision.bbox import ExtractBoundingBoxes, Iou, Nms

    class DummyTensor:
        shape = (1, 2, 3)

    t = DummyTensor()
    assert ExtractBoundingBoxes().infer_shape(t) == (1, 2, 3)
    assert Iou().infer_shape(t) == (1, 2, 3)
    assert Nms().infer_shape(t) == (1, 2, 3)
