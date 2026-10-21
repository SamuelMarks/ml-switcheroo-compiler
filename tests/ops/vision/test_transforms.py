"""Tests for test_transforms."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.transforms import elastic_transform, flip_left_right, flip_up_down, perspective_transform
from tests.testing_utils import as_callable, as_tensor

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


def test_vision_transforms_funcs(mocker: EagerTensorProtocol) -> None:
    """Test function test_vision_transforms_funcs.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.transforms._emit_shape_node", return_value="node")
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.vision.transforms.get_op")
    mock_op = mocker.MagicMock()
    setattr(mock_get_op, "return_value", mocker.MagicMock(return_value=mock_op))
    assert perspective_transform(t, t, t) == "node"
    assert elastic_transform(t, t) == "node"
    assert flip_left_right(t) == mock_op()
    assert flip_up_down(t) == mock_op()
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.transforms.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert as_tensor(getattr(perspective_transform(t, t, t), "config")).shape == (2, 3)
    assert as_tensor(getattr(elastic_transform(t, t), "config")).shape == (2, 3)
    assert as_tensor(getattr(flip_left_right(t), "config")).shape == (2, 3)
    assert as_tensor(getattr(flip_up_down(t), "config")).shape == (2, 3)


def test_vision_transforms_funcs_kwargs(mocker: EagerTensorProtocol) -> None:
    """Test function test_vision_transforms_funcs_kwargs.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.transforms._emit_shape_node", return_value="node")
    assert perspective_transform(t, t, t, config_obj="cfg") == "node"
    assert elastic_transform(t, t, config_obj="cfg") == "node"


def test_transforms_opdef_infer_shape() -> None:
    """Test function test_transforms_opdef_infer_shape."""
    import ml_switcheroo_compiler.ops.vision.frontend as frontend
    from ml_switcheroo_compiler.ops.vision.transforms import ElasticTransform

    assert frontend is not None

    class DummyTensor:
        """Test class DummyTensor."""

        shape = (1, 2, 3)

    t = DummyTensor()
    assert as_callable(ElasticTransform)().infer_shape(t) == (1, 2, 3)
