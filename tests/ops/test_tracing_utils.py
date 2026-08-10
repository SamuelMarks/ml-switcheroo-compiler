"""Tests for tracing utility functions."""

import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _get_tensor_ids


class MockData:
    def __init__(self, id_val="mock"):
        self.id = id_val


def test_get_tensor_ids():
    t1 = Tensor(MockData("t1"), TensorConfig((), DType.Float32, Device("cpu")))
    t2 = Tensor(MockData("t2"), TensorConfig((), DType.Float32, Device("cpu")))

    assert _get_tensor_ids(t1) == ["t1"]
    assert _get_tensor_ids([t1, t2]) == ["t1", "t2"]
    assert _get_tensor_ids((t1, t2)) == ["t1", "t2"]

    with pytest.raises(TypeError):
        _get_tensor_ids({"a": t1, "b": t2})

    with pytest.raises(TypeError):
        _get_tensor_ids(1)
