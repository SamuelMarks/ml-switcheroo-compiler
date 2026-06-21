import numpy as np
import pytest

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg import power_iteration


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax", "mlx"])
def test_power_iteration_convergence(backend_name):
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(backend=backend_name, eager_mode=True):
        try:
            backend = get_active_backend()
        except ValueError:
            pytest.skip(f"Backend {backend_name} not available")
        w_np = np.random.randn(2, 4, 4).astype(np.float32)
        w_data = backend.array(w_np)
        w_tensor = Tensor(w_data, TensorConfig(w_np.shape, DType.Float32, device))

        # SVD reference
        _, s, _ = np.linalg.svd(w_np)
        expected_sigma = s[..., 0]

        # Test 10 iterations
        v, u, sigma = power_iteration(w_tensor, num_iters=20)

        sigma_np = (
            backend.to_numpy(sigma.data) if hasattr(backend, "to_numpy") else np.array(sigma.data)
        )

        # In 20 iterations it should be reasonably close
        np.testing.assert_allclose(sigma_np, expected_sigma, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax", "mlx"])
def test_power_iteration_with_u(backend_name):
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(backend=backend_name, eager_mode=True):
        try:
            backend = get_active_backend()
        except ValueError:
            pytest.skip(f"Backend {backend_name} not available")
        w_np = np.random.randn(3, 3).astype(np.float32)
        u_np = np.ones((3, 1), dtype=np.float32)

        w_data = backend.array(w_np)
        u_data = backend.array(u_np)
        w_tensor = Tensor(w_data, TensorConfig(w_np.shape, DType.Float32, device))
        u_tensor = Tensor(u_data, TensorConfig(u_np.shape, DType.Float32, device))

        v, u, sigma = power_iteration(w_tensor, num_iters=2, u=u_tensor)

        assert list(v.shape) == [3]
        assert list(u.shape) == [3]
        assert list(sigma.shape) == []


def test_power_iteration_tracing():
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.tracer import _tracer

    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            w = Tensor("dummy_w", TensorConfig((3, 3), DType.Float32, device))
            u = Tensor("dummy_u", TensorConfig((3, 1), DType.Float32, device))

            v, u_out, sigma = power_iteration(w, num_iters=2, u=u)
            v2, u2, s2 = power_iteration(w, num_iters=5)
        finally:
            _tracer.stop_tracing()


def test_power_iteration_generator():
    from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
    from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
    from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
    from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n1 = IRNode(id="w", op_type="Input", inputs=[])
    n2 = IRNode(id="u", op_type="Input", inputs=[])
    n3 = IRNode(
        id="power_iter",
        op_type="PowerIteration",
        inputs=["w", "u"],
        attributes={"num_iters": 2},
    )
    g.nodes = {n1.id: n1, n2.id: n2, n3.id: n3}

    jax_code = JAXCodeGenerator(g).generate()
    assert "jax_power_iteration" in jax_code

    mlx_code = MLXCodeGenerator(g).generate()
    assert "mlx_power_iteration" in mlx_code

    np_code = NumpyGenerator(g).generate()
    assert "np_power_iteration" in np_code

    pt_code = PyTorchCodeGenerator(g).generate()
    assert "pt_power_iteration" in pt_code
