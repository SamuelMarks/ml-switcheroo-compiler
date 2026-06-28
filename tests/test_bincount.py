import pytest
import numpy as np
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.aliases.numpy_compat import bincount
from ml_switcheroo_compiler.ops.reductions import bincount as bincount_op
from ml_switcheroo_compiler.tracing.tracer import _tracer


def test_bincount_eager():
    config.eager_mode = True
    x = Tensor(np.array([0, 1, 1, 3]), TensorConfig((4,), DType.Int32, Device("cpu")))
    res = bincount(x)
    assert list(res.data) == [1, 2, 0, 1]


def test_bincount_tracing():
    config.eager_mode = False
    x = Tensor(np.array([0, 1, 1, 3]), TensorConfig((4,), DType.Int32, Device("cpu")))
    with pytest.raises(NotImplementedError):
        bincount(x)


def test_bincount_op():
    config.eager_mode = False
    x = Tensor(np.array([0, 1, 1, 3]), TensorConfig((4,), DType.Int32, Device("cpu")))
    graph = _tracer.start_tracing("test")
    res = bincount_op(x)
    assert res is not None
    assert len(graph.nodes) > 0
    assert any(n.op_type == "Bincount" for n in graph.nodes.values())
    _tracer.stop_tracing()
