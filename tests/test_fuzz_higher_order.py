import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import GradOptions, hessian, jacfwd, jacrev


def make_tensor(data):
    return Tensor(np.array(data, dtype=np.float32), TensorConfig(np.shape(data), DType.Float32, Device("cpu")))


@pytest.mark.parametrize("backend_name", ["numpy", "mlx", "pytorch", "jax"])
@settings(max_examples=10, deadline=None)
@given(
    val1=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_higher_order_equivalence(backend_name, val1):
    # If the backend is not installed, skip
    try:
        with ConfigContext(backend=backend_name):
            backend = get_active_backend()
            backend.asarray([1.0])
    except Exception:
        pytest.skip(f"Backend {backend_name} not available")

    def f(x):
        # f(x) = x^3 + x
        x3 = ops.multiply(ops.multiply(x, x), x)
        return ops.add(x3, x)

    t1_data = [val1, val1 + 1.0]
    opts = GradOptions()
    opts.argnums = 0

    with ConfigContext(backend="numpy", eager_mode=True):
        # We use numpy as the source of truth
        jf_np = jacfwd(f, options=opts)(Tensor(get_active_backend().asarray(t1_data), TensorConfig((2,), DType.Float32, Device("cpu"))))
        jr_np = jacrev(f, options=opts)(Tensor(get_active_backend().asarray(t1_data), TensorConfig((2,), DType.Float32, Device("cpu"))))
        h_np = hessian(f, options=opts)(Tensor(get_active_backend().asarray(t1_data), TensorConfig((2,), DType.Float32, Device("cpu"))))

    with ConfigContext(backend=backend_name, eager_mode=True):
        jf_b = jacfwd(f, options=opts)(Tensor(get_active_backend().asarray(t1_data), TensorConfig((2,), DType.Float32, Device("cpu"))))
        jr_b = jacrev(f, options=opts)(Tensor(get_active_backend().asarray(t1_data), TensorConfig((2,), DType.Float32, Device("cpu"))))
        h_b = hessian(f, options=opts)(Tensor(get_active_backend().asarray(t1_data), TensorConfig((2,), DType.Float32, Device("cpu"))))

    np.testing.assert_allclose(get_active_backend().asarray(jf_np), get_active_backend().asarray(jf_b), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(get_active_backend().asarray(jr_np), get_active_backend().asarray(jr_b), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(get_active_backend().asarray(h_np), get_active_backend().asarray(h_b), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("backend_name", ["numpy", "mlx", "pytorch", "jax"])
@settings(max_examples=10, deadline=None)
@given(val1=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
def test_higher_order_control_flow(backend_name, val1):
    try:
        with ConfigContext(backend=backend_name):
            backend = get_active_backend()
            backend.asarray([1.0])
    except Exception:
        pytest.skip(f"Backend {backend_name} not available")

    def f_cf(x):
        x3 = ops.multiply(ops.multiply(x, x), x)
        cond_res = ops.cond(ops.greater(ops.sum(x), 0.0), lambda: ops.add(x3, x), lambda: ops.subtract(x3, x))
        loop_res = ops.while_loop(lambda v: ops.less(ops.sum(v), 3.0), lambda v: ops.add(v, 1.0), cond_res)
        return loop_res

    t1_data = [val1, val1 + 1.0]

    with ConfigContext(backend="numpy", eager_mode=True):
        res_np = f_cf(Tensor(get_active_backend().asarray(t1_data), TensorConfig((2,), DType.Float32, Device("cpu"))))

    with ConfigContext(backend=backend_name, eager_mode=True):
        res_b = f_cf(Tensor(get_active_backend().asarray(t1_data), TensorConfig((2,), DType.Float32, Device("cpu"))))

    np.testing.assert_allclose(get_active_backend().asarray(res_np), get_active_backend().asarray(res_b), rtol=1e-3, atol=1e-3)
