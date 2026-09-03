import gc
from unittest.mock import MagicMock

import numpy as np
import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.utils import _compute_grad_and_value, _convert_to_tensors, _get_inputs_dict, value_and_grad_wrt_vars


def test_value_and_grad_wrt_vars():
    def dummy_fun(x):
        return x * 2

    wrapped = value_and_grad_wrt_vars(dummy_fun)
    val, grads = wrapped(5)
    assert val == 10
    assert isinstance(grads, dict)


def test_get_inputs_dict():
    # To hit 200 we need a tensor tracked by gc
    class MockTensor(Tensor):
        @property
        def data(self):
            m = MagicMock()
            m.id = "my_node"
            m.concrete_value = 5.0
            return m

    t = MockTensor(np.array(5.0), TensorConfig(shape=(1,), dtype=DType.Float32, device=Device("cpu")))

    # We must ensure gc sees this
    objs = gc.get_objects()

    graph = MagicMock()
    graph.nodes = {"my_node": MagicMock()}

    res = _get_inputs_dict(graph)
    assert "my_node" in res
    # Keep reference
    assert t is not None


def test_convert_to_tensors_types():
    primals = [np.array(5.0, dtype=np.float64), np.array(5, dtype=np.int32), np.array(True, dtype=np.bool_)]
    res = _convert_to_tensors(primals)
    assert res[0].config.dtype == DType.Float64
    assert res[1].config.dtype == DType.Int32
    assert res[2].config.dtype == DType.Bool


def test_compute_grad_and_value_argnums():
    def fun(x, y):
        return x + y

    options = MagicMock()
    options.has_aux = False

    # tuple argnums
    options.argnums = (0, 1)

    with pytest.MonkeyPatch.context() as m:
        m.setattr("ml_switcheroo_compiler.grad.jvp_vjp.vjp", lambda f, *a, **kw: (f(*a), lambda cot: (cot, cot)))
        val, grad = _compute_grad_and_value(fun, options, (1.0, 2.0))
        assert val == 3.0
        assert isinstance(grad, tuple)

    # None/other argnums
    options.argnums = None
    with pytest.MonkeyPatch.context() as m:
        m.setattr("ml_switcheroo_compiler.grad.jvp_vjp.vjp", lambda f, *a, **kw: (f(*a), lambda cot: (cot, cot)))
        val, grad = _compute_grad_and_value(fun, options, (1.0, 2.0))
        assert val == 3.0
