"""Tests for test_bbox."""

from __future__ import annotations

import typing

import pytest
from pytest_mock import MockerFixture

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.bbox import (
    _extract_bounding_boxes_eager,
    crop,
    crop_and_resize,
    crop_images,
    draw_bounding_boxes,
    extract_bounding_boxes,
    extract_patches,
    pad_images,
    pad_to_bounding_box,
)
from tests.testing_utils import MockTestStub, ensure_tensor, extract_test_shape

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


def test_vision_bbox_funcs(mocker: MockerFixture) -> None:
    """Test function test_vision_bbox_funcs.

    Args:
        mocker: Test parameter `mocker`.
    """
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
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)), raising=False)
    pytest.MonkeyPatch().setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)), raising=False)
    assert ensure_tensor(crop_and_resize(t, t, t, (10, 10))).config.shape == (2, 3)
    assert ensure_tensor(extract_bounding_boxes(t, t, t, crop_size=10)).config.shape == (
        2,
        3,
    )
    assert ensure_tensor(crop(t, 1, 1, 10, 10)).config.shape == (2, 3)
    assert ensure_tensor(pad_to_bounding_box(t, 1, 1, 10, 10)).config.shape == (2, 3)
    assert ensure_tensor(draw_bounding_boxes(t, t, t)).config.shape == (2, 3)
    assert ensure_tensor(crop_images(t, (1, 1, 1, 1))).config.shape == (2, 3)
    assert ensure_tensor(extract_patches(t, (10, 10))).config.shape == (2, 3)
    assert ensure_tensor(pad_images(t, (1, 1, 1, 1), (10, 10))).config.shape == (2, 3)


def test_extract_bounding_boxes_eager_extra(mocker: MockerFixture) -> None:
    """Test function test_extract_bounding_boxes_eager_extra.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)), raising=False)
    pytest.MonkeyPatch().setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)), raising=False)
    res = _extract_bounding_boxes_eager(t, t, t, None)
    assert tuple(extract_test_shape(res)) == (2, 3)


def test_extract_bounding_boxes_kwargs(mocker: MockerFixture) -> None:
    """Test function test_extract_bounding_boxes_kwargs.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.bbox._emit_shape_node", return_value="node")
    assert extract_bounding_boxes(t, t, t, config_obj="cfg") == "node"
    assert extract_bounding_boxes(t, t, t, crop_size=(10, 10)) == "node"


def test_bbox_opdef_infer_shape() -> None:
    """Test function test_bbox_opdef_infer_shape."""
    from ml_switcheroo_compiler.ops.vision.bbox import ExtractBoundingBoxes, Iou, Nms

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        shape = (1, 2, 3)

    t = DummyTensor()
    assert ExtractBoundingBoxes().infer_shape(t) == (1, 2, 3)
    assert Iou().infer_shape(t) == (1, 2, 3)
    assert Nms().infer_shape(t) == (1, 2, 3)
