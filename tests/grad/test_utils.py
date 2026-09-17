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


def test_grad_utils_extra():
    """Test edge cases in grad utils including scalar checking, proxy values, and type conversion."""
    from unittest.mock import patch

    import numpy as np
    import pytest

    from ml_switcheroo_compiler.core.errors import SwitcherooError
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.utils import _check_scalar, _get_concrete_val, _get_inputs_dict, _to_original_type

    class MockTensor:
        shape = (1, "a")

    with pytest.raises(SwitcherooError, match="backward"):
        _check_scalar(MockTensor())

    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    class MockTensorProxy:
        _data = ProxyTensor(None, (1,))

    assert _get_concrete_val(MockTensorProxy()) is None

    class MockProxy(ProxyTensor):
        def __init__(self):
            pass

    mock_proxy = MockProxy()
    mock_proxy.concrete_value = 5.0
    mock_tensor_proxy2 = MockTensorProxy()
    mock_tensor_proxy2._data = mock_proxy
    assert _get_concrete_val(mock_tensor_proxy2) == 5.0

    class MockGraph:
        inputs = ["in1"]
        nodes = {}

    with pytest.raises(ValueError, match="Missing input"):
        _get_inputs_dict(MockGraph())

    t = Tensor(1.0, TensorConfig((1,), "float32", "cpu"))
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.asarray.return_value = np.array([1.0], dtype=np.float64)
        res = _to_original_type(1.0, t)
        assert getattr(res, "dtype", None) == DType.Float64

        mock_backend.return_value.asarray.return_value = np.array([1], dtype=np.int32)
        res = _to_original_type(1, t)
        assert getattr(res, "dtype", None) == DType.Int32

        mock_backend.return_value.asarray.return_value = np.array([True], dtype=bool)
        res = _to_original_type(True, t)
        assert getattr(res, "dtype", None) == DType.Bool
