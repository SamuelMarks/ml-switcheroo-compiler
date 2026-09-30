"""Tests for test_tensor_array_branches."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.tensor_array import TensorArray


def test_tensor_array_read_existing_value_branch() -> None:
    """Test TensorArray.read when value at index is not None."""
    ta = TensorArray(size=4, element_shape=(2, 2), dtype="float32")
    idx_tensor = Tensor(np.array(0, dtype=np.int32), TensorConfig((), DType.Int32, Device("cpu")))
    val_tensor = Tensor(np.ones((2, 2), dtype=np.float32), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    ta.write(idx_tensor, val_tensor)

    res = ta.read(idx_tensor)
    assert res.shape == (2, 2)
    np.testing.assert_array_equal(res.data, np.ones((2, 2), dtype=np.float32))
