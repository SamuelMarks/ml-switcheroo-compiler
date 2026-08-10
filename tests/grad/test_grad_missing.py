import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import GradOptions, _generate_fallback_input, _to_original_type, checkpoint, value_and_grad
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_checkpoint_eager():
    config.eager_mode = True

    def my_func(x):
        return x * 2

    wrapped = checkpoint(my_func)
    assert wrapped(3) == 6
    config.eager_mode = False


def test_checkpoint_not_tracing():
    global_tracing_state.is_tracing = False

    def my_func(x):
        return x * 2

    wrapped = checkpoint(my_func)
    assert wrapped(4) == 8


def test_checkpoint_infer_dtype():
    global_tracing_state.is_tracing = True

    def my_func(x):
        return x

    wrapped = checkpoint(my_func)
    t = Tensor(np.array([1, 2], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    try:
        wrapped(t)
    except Exception:
        pass
    global_tracing_state.is_tracing = False


def test_generate_fallback_input_value_error():
    class MockNode:
        shape_metadata = ("not_an_int", 2)
        attributes = {"dtype": "float32"}

    class MockGraph:
        nodes = {"node_1": MockNode()}

    res = _generate_fallback_input(MockGraph(), "node_1")
    assert res.shape == (1, 2)


def test_reconstruct_output_float64():
    t_orig = Tensor(np.array(1.0, dtype=np.float64), TensorConfig((), DType.Float64, Device("cpu")))
    res = _to_original_type(np.array(2.0, dtype=np.float64), t_orig)
    assert res.dtype == DType.Float64
    assert res.item() == 2.0


def test_grad_and_value_has_aux():
    def my_func(x):
        return x * x, x

    opt = GradOptions(has_aux=True)
    wrapped = value_and_grad(my_func, opt)
    t = Tensor(np.array(2.0, dtype=np.float32), TensorConfig((), DType.Float32, Device("cpu")))
    wrapped(t)
