"""Tests for test_mixing."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vision.mixing import cutmix, mixup
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

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


def test_vision_mixing_funcs(mocker: EagerTensorProtocol) -> None:
    """Test function test_vision_mixing_funcs.

    Args:
        mocker: Test parameter `mocker`.
    """
    t1 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    t2 = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.ops.vision.mixing._emit_shape_node", return_value="node")
    assert mixup(t1, t2) == "node"
    assert cutmix(t1, t2) == "node"
    config.eager_mode = True
    mock_backend = mocker.patch("ml_switcheroo_compiler.ops.vision.mixing.get_active_backend").return_value
    ensure_mutable(mock_backend.execute_op).return_value = ensure_eager_tensor(MockTensor((2, 3)))
    ensure_mutable(mock_backend.array).side_effect = ensure_eager_tensor(lambda x: MockTensor((2, 3)))
    assert ensure_eager_tensor(ensure_eager_tensor(mixup(t1, t2)).config).shape == (2, 3)
    assert ensure_eager_tensor(ensure_eager_tensor(cutmix(t1, t2)).config).shape == (2, 3)
