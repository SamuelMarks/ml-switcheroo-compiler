"""Tests for test_affine."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.affine import affine_generator, affine_grid, affine_transform, grid_sample, random_crop, random_elastic_transform, random_flip, random_perspective, random_rotation, random_shear, random_translation, random_zoom
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


def test_vision_affine_funcs(mocker: EagerTensorProtocol) -> None:
    """Test function test_vision_affine_funcs.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.affine._emit_shape_node", return_value="node")
    assert affine_transform(t, t) == "node"
    assert affine_generator(1, t, t, t) == "node"
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.vision.affine.get_op")
    mock_op = mocker.MagicMock()
    setattr(mock_get_op, "return_value", mocker.MagicMock(return_value=mock_op))
    assert random_flip(t) == mock_op()
    assert random_rotation(t, 0.1) == "node"
    assert random_crop(t, (1, 1)) == mock_op()
    assert random_zoom(t, (0.5, 0.5)) == "node"
    assert random_translation(t, 0.1, 0.1) == "node"
    assert random_shear(t, 0.1, 0.1) == "node"
    assert random_perspective(t, 0.1) == "node"
    assert random_elastic_transform(t, 0.1, 0.1) == "node"
    assert affine_grid(t, (2, 3)) == "node"
    assert grid_sample(t, t) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.affine.get_active_backend").return_value
    setattr(mock_backend.execute_op, "return_value", MockTensor((2, 3)))
    setattr(mock_backend.array, "side_effect", lambda x: MockTensor((2, 3)))
    assert as_tensor(getattr(affine_transform(t, t), "config")).shape == (2, 3)
    assert as_tensor(getattr(affine_generator(1, t, t, t), "config")).shape == (2, 3)
    assert as_tensor(getattr(random_flip(t), "config")).shape == (2, 3)
    assert as_tensor(getattr(random_rotation(t, 0.1), "config")).shape == (2, 3)
    assert as_tensor(getattr(random_crop(t, (1, 1)), "config")).shape == (2, 3)
    assert as_tensor(getattr(random_zoom(t, (0.5, 0.5)), "config")).shape == (2, 3)
    assert as_tensor(getattr(random_translation(t, 0.1, 0.1), "config")).shape == (2, 3)
    assert as_tensor(getattr(random_shear(t, 0.1, 0.1), "config")).shape == (2, 3)
    assert as_tensor(getattr(random_perspective(t, 0.1), "config")).shape == (2, 3)
    assert as_tensor(getattr(random_elastic_transform(t, 0.1, 0.1), "config")).shape == (2, 3)
    assert as_tensor(getattr(affine_grid(t, (2, 3)), "config")).shape == (2, 3)
    assert as_tensor(getattr(grid_sample(t, t), "config")).shape == (2, 3)


def test_affine_opdef_infer_shape() -> None:
    """Test function test_affine_opdef_infer_shape."""
    from ml_switcheroo_compiler.ops.vision.affine import AffineGenerator, AffineGrid, AffineTransform, PerspectiveTransform

    class DummyTensor:
        """Test class DummyTensor."""

        shape = (1, 2, 3)

    t = DummyTensor()
    assert as_callable(AffineGenerator)().infer_shape(t) == (1, 2, 3)
    assert as_callable(AffineGrid)().infer_shape(t) == (1, 2, 3)
    assert as_callable(AffineTransform)().infer_shape(t) == (1, 2, 3)
    assert as_callable(PerspectiveTransform)().infer_shape(t) == (1, 2, 3)
