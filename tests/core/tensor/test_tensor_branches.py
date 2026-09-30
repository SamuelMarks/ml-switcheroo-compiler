"""Tests for test_tensor_branches."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.shape_system import SymNode


def test_tensor_symnode_and_shape_tuple_branches() -> None:
    """Test Tensor with SymNode dimension and TensorConfig with already tuple shape."""
    sym = SymNode()
    cfg_tuple = TensorConfig(shape=(sym, 10), dtype=DType.Float32, device=Device("cpu"))
    assert isinstance(cfg_tuple.shape, tuple)

    t = Tensor(data=np.zeros((1, 10)), config=cfg_tuple)
    first_dim = t.shape[0]
    assert isinstance(first_dim, SymNode)
    assert t.shape[1] == 10
