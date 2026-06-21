"""Tests for extra methods in Tensor class."""

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_tensor_extra_methods() -> None:
    """Test function."""
    config.eager_mode = True
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", Device("cpu")))

    t_contig = t.contiguous()
    assert t_contig is t

    t_view = t.view(4)
    assert t_view.shape == (4,)

    t_view2 = t.view([4])
    assert t_view2.shape == (4,)

    t_detach = t.detach()
    assert t_detach.shape == (2, 2)
    assert t_detach is not t

    t_scalar = Tensor(np.array(5.0), TensorConfig((), "float32", Device("cpu")))
    assert t_scalar.item() == 5.0

    # backward just passes, let's call it
    t.backward()


def test_tensor_item_non_tensor() -> None:
    """Test function."""
    t_scalar = Tensor(np.array(5.0), TensorConfig((), "float32", Device("cpu")))
    # Mock eval to return scalar
    from unittest.mock import patch

    with patch.object(t_scalar, "eval", return_value=np.array(10.0)):
        assert t_scalar.item() == 10.0
