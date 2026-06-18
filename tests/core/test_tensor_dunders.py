"""Unit tests for verifying Tensor dunder (magic) methods and operations.

This module contains test cases to ensure that the Tensor class correctly implements and
executes various Python magic methods (dunders) such as arithmetic, bitwise, comparison,
unary, indexing, and type conversion operations in eager mode.
"""

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor


def test_tensor_dunders() -> None:
    """Verifies the correct execution of various magic (dunder) methods on the Tensor.

    class

    This test runs in eager mode and exercises arithmetic, bitwise, comparison,
    unary, indexing, in-place modification, and type conversion operations
    on Tensor instances to ensure they execute without errors

    Returns:
    None
    """
    config.eager_mode = True
    d = Device(DeviceType.CPU, 0)
    t1 = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, d)
    t2 = Tensor(np.array([4, 5, 6]), (3,), DType.Int32, d)
    t3 = Tensor(np.array([1]), (1,), DType.Int32, d)
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
    config.enable_in_place = True
    t1[0] = 5
    config.enable_in_place = False
    _ = np.array(t1)


def test_tensor_dunders_extra() -> None:
    """Tests extra tensor dunders for coverage."""
    import numpy as np

    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor

    t1 = Tensor(np.array(1), (), DType.Float32, Device(DeviceType.CPU), requires_grad=True)
    t2 = Tensor(np.array(2), (), DType.Float32, Device(DeviceType.CPU))

    assert t1.requires_grad

    # math
    assert (t1 + 1).data == 2
    assert (t1 - 1).data == 0
    assert (t1 * 2).data == 2
    assert (t2 / 2).data == 1
    assert (t2 // 2).data == 1
    assert (t2 % 2).data == 0
    assert (t2**2).data == 4

    # reverse math
    assert (1 + t1).data == 2
    assert (2 - t1).data == 1
    assert (2 * t1).data == 2
    assert (2 / t2).data == 1

    # bitwise
    t3 = Tensor(np.array(1), (), DType.Int32, Device(DeviceType.CPU))
    t4 = Tensor(np.array(2), (), DType.Int32, Device(DeviceType.CPU))
    assert (t3 & t4).data == 0
    assert (t3 | t4).data == 3
    assert (t3 ^ t4).data == 3
    assert (t3 << t4).data == 4
    assert (t4 >> t3).data == 1
    assert (~t3).data == -2

    # logic
    assert (t1 < t2).data
    assert not (t1 > t2).data
    assert (t1 <= t2).data
    assert not (t1 >= t2).data
    assert not (t1 == t2).data
    assert (t1 != t2).data


def test_tensor_errors() -> None:
    """Tests tensor errors."""
    import numpy as np
    import pytest

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor

    t = Tensor(np.array([1, 2]), (2,), DType.Float32, Device(DeviceType.CPU))
    with pytest.raises(ValueError):
        bool(t)

    config.eager_mode = False
    with pytest.raises(RuntimeError):
        t[0]
    with pytest.raises(TypeError):
        t[0] = 1
    config.eager_mode = True


def test_tensor_tracing_eval() -> None:
    """Tests tensor eval in tracing."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    config.eager_mode = False
    g = _tracer.start_tracing()
    pt = ProxyTensor("test_id", (1,), DType.Float32)
    t = Tensor(pt, (1,), DType.Float32, Device(DeviceType.CPU))
    t.eval()
    assert "test_id" in g.outputs
    t.eval()  # test already in outputs
    _tracer.stop_tracing()
    config.eager_mode = True


def test_tensor_coverage_more() -> None:
    """Tests more tensor coverage."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    t1 = Tensor(np.array(1), (), DType.Float32, Device(DeviceType.CPU))
    assert (+t1).data == 1

    pt = ProxyTensor("test_id", (1,), DType.Float32)
    t_proxy = Tensor(pt, (1,), DType.Float32, Device(DeviceType.CPU))
    assert np.array(t_proxy).shape == (1,)

    config.eager_mode = True
    t_tuple = Tensor(np.array([1, 2]), (2,), DType.Float32, Device(DeviceType.CPU))
    assert t_tuple[(0,)].data == 1

    t_tuple2 = Tensor(np.array([[1, 2], [3, 4]]), (2, 2), DType.Float32, Device(DeviceType.CPU))
    assert t_tuple2[0, 0].data == 1


def test_tensor_coverage_even_more() -> None:
    """Tests even more tensor coverage."""
    import numpy as np

    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor

    t1 = Tensor(np.array(1), (), DType.Float32, Device(DeviceType.CPU))
    assert (-t1).data == -1

    t_tuple = Tensor(np.array([1, 2]), (2,), DType.Float32, Device(DeviceType.CPU))

    # test __getitem__ with Tensor as key
    idx = Tensor(np.array(0), (), DType.Int32, Device(DeviceType.CPU))
    assert t_tuple[idx].data == 1
