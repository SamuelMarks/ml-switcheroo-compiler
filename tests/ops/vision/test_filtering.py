"""Tests for test_filtering."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.filtering import gaussian_blur, iou, median_filter, non_max_suppression, random_gaussian_blur, random_sharpness, sharpen

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


def test_vision_filtering_funcs(mocker: EagerTensorProtocol) -> None:
    """Test function test_vision_filtering_funcs.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.filtering._emit_shape_node", return_value="node")
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.vision.filtering.get_op")
    mock_op = mocker.MagicMock()
    setattr(mock_get_op, "return_value", mocker.MagicMock(return_value=mock_op))
    assert gaussian_blur(t) == "node"
    assert gaussian_blur(t, kernel_size=3, sigma=1) == "node"
    assert median_filter(t, 3) == "node"
    assert iou(t, t) == "node"
    assert non_max_suppression(t, t, 1) == "node"
    assert sharpen(t) == mock_op()
    assert random_gaussian_blur(t, 3, 1) == "node"
    assert random_sharpness(t, 1.0) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.filtering.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert typing.cast("EagerTensorProtocol", getattr(gaussian_blur(t), "config")).shape == (2, 3)
    assert typing.cast("EagerTensorProtocol", getattr(median_filter(t, 3), "config")).shape == (2, 3)
    assert typing.cast("EagerTensorProtocol", getattr(iou(t, t), "config")).shape == (2, 3)
    assert typing.cast("EagerTensorProtocol", getattr(non_max_suppression(t, t, 1), "config")).shape == (2, 3)


def test_vision_filtering_kwargs(mocker: EagerTensorProtocol) -> None:
    """Test function test_vision_filtering_kwargs.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.filtering._emit_shape_node", return_value="node")
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.vision.filtering.get_op")
    mock_op = mocker.MagicMock()
    setattr(mock_get_op, "return_value", mocker.MagicMock(return_value=mock_op))
    assert gaussian_blur(t, config_obj="cfg") == "node"


def test_vision_filtering_eager_extra(mocker: EagerTensorProtocol) -> None:
    """Test function test_vision_filtering_eager_extra.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.filtering.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert typing.cast("EagerTensorProtocol", getattr(sharpen(t), "config")).shape == (2, 3)
    assert typing.cast("EagerTensorProtocol", getattr(random_gaussian_blur(t, 3, 1), "config")).shape == (2, 3)
    assert typing.cast("EagerTensorProtocol", getattr(random_sharpness(t, 1.0), "config")).shape == (2, 3)


def test_median_filter_tuple(mocker: EagerTensorProtocol) -> None:
    """Test function test_median_filter_tuple.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.filtering._emit_shape_node", return_value="node")
    assert median_filter(t, kernel_size=(3, 3)) == "node"
