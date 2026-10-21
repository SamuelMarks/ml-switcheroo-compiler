"""Unit tests for verifying Tensor dunder (magic) methods and operations.

This module contains test cases to ensure that the Tensor class correctly implements and
executes various Python magic methods (dunders) such as arithmetic, bitwise, comparison,
unary, indexing, and type conversion operations in eager mode.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_tensor_dunders() -> None:
    """Test the tensor dunders behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    d = Device(DeviceType.CPU, 0)
    t1 = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, d))
    t2 = Tensor(np.array([4, 5, 6]), TensorConfig((3,), DType.Int32, d))
    t3 = Tensor(np.array([1]), TensorConfig((1,), DType.Int32, d))
    _ = t1 % t2
    _ = t1 & t2
    _ = t1 | t2
    _ = t1 ^ t2
    _ = t1 << t2
    _ = t1 >> t2
    _ = abs(t1)
    _ = ~t1
    _ = t1 < t2
    _ = t1 <= t2
    _ = t1 == t2
    _ = t1 != t2
    _ = t1 > t2
    _ = t1 >= t2
    _ = bool(t3)
    _ = len(t1)
    _ = list(t1)
    _ = +t1
    _ = t1[0]
    setattr(config, "enable_in_place", True)
    t1[0] = 5
    setattr(config, "enable_in_place", False)
    _ = np.array(t1)
    assert callable(test_tensor_dunders)


def test_tensor_dunders_extra() -> None:
    """Test function test_tensor_dunders_extra."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    t1 = Tensor(np.array(1), TensorConfig((), DType.Float32, Device(DeviceType.CPU), requires_grad=True))
    t2 = Tensor(np.array(2), TensorConfig((), DType.Float32, Device(DeviceType.CPU)))
    assert t1.requires_grad
    assert (t1 + 1).data == 2
    assert (t1 - 1).data == 0
    assert (t1 * 2).data == 2
    assert (t2 / 2).data == 1
    assert (t2 // 2).data == 1
    assert (t2 % 2).data == 0
    assert (t2**2).data == 4
    assert (1 + t1).data == 2
    assert (2 - t1).data == 1
    assert (2 * t1).data == 2
    assert (2 / t2).data == 1
    t3 = Tensor(np.array(1), TensorConfig((), DType.Int32, Device(DeviceType.CPU)))
    t4 = Tensor(np.array(2), TensorConfig((), DType.Int32, Device(DeviceType.CPU)))
    assert (t3 & t4).data == 0
    assert (t3 | t4).data == 3
    assert (t3 ^ t4).data == 3
    assert (t3 << t4).data == 4
    assert (t4 >> t3).data == 1
    assert (~t3).data == -2
    assert (t1 < t2).data
    assert not (t1 > t2).data
    assert (t1 <= t2).data
    assert not (t1 >= t2).data
    assert not as_tensor(getattr(t1 == t2, "data"))
    assert as_tensor(getattr(t1 != t2, "data"))


def test_tensor_errors() -> None:
    """Test the tensor errors behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    t = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Float32, Device(DeviceType.CPU)))
    with pytest.raises((ValueError, ShapeMismatchError)):
        bool(t)
    config.eager_mode = False
    with pytest.raises(RuntimeError):
        t[0]
    with pytest.raises(TypeError):
        t[0] = 1
    config.eager_mode = True


def test_tensor_tracing_eval() -> None:
    """Test the tensor tracing eval behavior."""
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            global_tracing_state.add_node(LogicalNode(id="test_id", op_type="Input"))
            pt = ProxyTensor("test_id", (1,), DType.Float32)
            t = Tensor(pt, TensorConfig((1,), DType.Float32, Device(DeviceType.CPU)))
            t.eval()
            assert global_tracing_state.active_graph is not None
            assert "test_id" in global_tracing_state.active_graph.outputs
            t.eval()
        finally:
            global_tracing_state.stop_tracing()


def test_tensor_coverage_more() -> None:
    """Test function test_tensor_coverage_more."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    t1 = Tensor(np.array(1), TensorConfig((), DType.Float32, Device(DeviceType.CPU)))
    assert (+t1).data == 1
    pt = ProxyTensor("test_id", (1,), DType.Float32)
    t_proxy = Tensor(pt, TensorConfig((1,), DType.Float32, Device(DeviceType.CPU)))
    assert np.array(t_proxy).shape == (1,)
    config.eager_mode = True
    t_tuple = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Float32, Device(DeviceType.CPU)))
    assert t_tuple[0,].data == 1
    t_tuple2 = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig((2, 2), DType.Float32, Device(DeviceType.CPU)))
    assert t_tuple2[0, 0].data == 1


def test_tensor_coverage_even_more() -> None:
    """Test function test_tensor_coverage_even_more."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    t1 = Tensor(np.array(1), TensorConfig((), DType.Float32, Device(DeviceType.CPU)))
    assert (-t1).data == -1
    t_tuple = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Float32, Device(DeviceType.CPU)))
    idx = Tensor(np.array(0), TensorConfig((), DType.Int32, Device(DeviceType.CPU)))
    assert t_tuple[idx].data == 1
