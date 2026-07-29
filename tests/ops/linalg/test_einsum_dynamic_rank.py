import numpy as np
import pytest

import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


@pytest.mark.parametrize("backend_name", ["numpy", "mlx", "pytorch", "jax"])
def test_einsum_ellipsis(backend_name):
    try:
        with ConfigContext(backend=backend_name):
            backend = get_active_backend()
            backend.asarray([1.0])
    except Exception:
        pytest.skip(f"Backend {backend_name} not available")

    a_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b_np = np.random.randn(3, 5, 6).astype(np.float32)

    with ConfigContext(backend="numpy", eager_mode=True):
        t_a = Tensor(a_np, TensorConfig(a_np.shape, DType.Float32, Device("cpu")))
        t_b = Tensor(b_np, TensorConfig(b_np.shape, DType.Float32, Device("cpu")))
        out_ref = ops.einsum("...ab,...bc->...ac", t_a, t_b)

    with ConfigContext(backend=backend_name, eager_mode=True):
        t_a_b = Tensor(get_active_backend().asarray(a_np), TensorConfig(a_np.shape, DType.Float32, Device("cpu")))
        t_b_b = Tensor(get_active_backend().asarray(b_np), TensorConfig(b_np.shape, DType.Float32, Device("cpu")))
        out_b = ops.einsum("...ab,...bc->...ac", t_a_b, t_b_b)

    np.testing.assert_allclose(get_active_backend().asarray(out_ref), get_active_backend().asarray(out_b), rtol=1e-5, atol=1e-5)
