import numpy as np
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random.distributions_discrete import binomial
from ml_switcheroo_compiler.tracing.tracer import _tracer, ProxyTensor


def test_binomial_eager():
    config.eager_mode = True
    key = Tensor(np.array([0, 0]), TensorConfig((2,), DType.UInt32, Device("cpu")))
    n = Tensor(np.array(10), TensorConfig((), DType.Int32, Device("cpu")))
    p = Tensor(np.array(0.5), TensorConfig((), DType.Float32, Device("cpu")))
    res = binomial(key, n, p, shape=(2,), dtype=DType.Int32)
    assert res.shape == (2,)
    assert res.dtype == DType.Int32


def test_binomial_tracing():
    config.eager_mode = False
    graph = _tracer.start_tracing("test")
    key = Tensor(ProxyTensor(id="key", shape=(2,)), TensorConfig((2,), DType.UInt32, Device("cpu")))
    n = Tensor(ProxyTensor(id="n", shape=()), TensorConfig((), DType.Int32, Device("cpu")))
    p = Tensor(ProxyTensor(id="p", shape=()), TensorConfig((), DType.Float32, Device("cpu")))
    res = binomial(key, n, p, shape=(2,), dtype=DType.Int32)
    assert res is not None
    assert len(graph.nodes) > 0
    assert any(n.op_type == "RandomBinomial" for n in graph.nodes.values())
    _tracer.stop_tracing()
