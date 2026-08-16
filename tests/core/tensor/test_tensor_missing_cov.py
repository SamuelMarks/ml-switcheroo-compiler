"""Tests for missing coverage in tensor.py."""

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_tensor_config_list_shape() -> None:
    """Test initializing TensorConfig with list shape."""
    config = TensorConfig(shape=[2, 2], dtype=DType("float32"), device=Device("cpu"))
    assert config.shape == (2, 2)


def test_tensor_view() -> None:
    """Test tensor view with different shape args."""
    config = TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu"))
    t = Tensor(42, config=config)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.registry.get_frontend") as mock_frontend:
        mock_frontend.return_value = lambda x, s: x
        res = t.view(2, 2)
        assert res is t

        res2 = t.view([2, 2])
        assert res2 is t
