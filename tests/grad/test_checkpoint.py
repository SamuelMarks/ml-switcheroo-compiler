"""Unit tests for activation checkpointing and rematerialization."""

import numpy as np

import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import checkpoint, grad, remat, value_and_grad
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function


def test_checkpoint_activation_eviction_and_memory():
    """Verify that intermediate activations are evicted from the outer tape during checkpointing."""

    def complex_forward(x):
        """Forward pass with multiple intermediate steps."""
        h1 = ops.multiply(x, x)
        h2 = ops.add(h1, x)
        h3 = ops.multiply(h2, h1)
        h4 = ops.subtract(h3, x)
        return ops.multiply(h4, h2)

    cp_forward = checkpoint(complex_forward)

    t = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    with ConfigContext(eager_mode=False):
        block_standard = _trace_function(complex_forward, (t,), "std")
        block_checkpoint = _trace_function(cp_forward, (t,), "cp")

    std_nodes = [n.op_type for n in (block_standard.nodes if isinstance(block_standard.nodes, list) else block_standard.nodes.values())]
    cp_nodes = [n.op_type for n in (block_checkpoint.nodes if isinstance(block_checkpoint.nodes, list) else block_checkpoint.nodes.values())]

    # Standard forward keeps all intermediate operations in the tape
    assert len(std_nodes) > len(cp_nodes)
    assert "Checkpoint" in cp_nodes
    # Intermediate Multiply, Add, Subtract operations are not present in the outer checkpoint tape
    assert cp_nodes.count("Multiply") == 0
    assert cp_nodes.count("Add") == 0
    assert cp_nodes.count("Subtract") == 0


def test_checkpoint_gradient_exact_numerical_match():
    """Verify exact float-for-float equivalence between standard autodiff and checkpointed backward pass."""

    def f(x):
        """Nonlinear function."""
        y = ops.multiply(x, x)
        z = ops.add(y, ops.multiply(x, 3.0))
        return ops.multiply(z, ops.subtract(y, 1.0))

    f_cp = remat(f)

    arr = np.array([2.5], dtype=np.float32)
    t1 = Tensor(arr.copy(), TensorConfig((1,), DType.Float32, Device("cpu")))
    t2 = Tensor(arr.copy(), TensorConfig((1,), DType.Float32, Device("cpu")))

    grad_fn_std = grad(f)
    grad_fn_cp = grad(f_cp)

    g_std = grad_fn_std(t1)
    g_cp = grad_fn_cp(t2)

    val_std = get_active_backend().asarray(g_std)
    val_cp = get_active_backend().asarray(g_cp)

    np.testing.assert_allclose(val_std, val_cp, rtol=1e-5, atol=1e-5)


def test_checkpoint_value_and_grad_parity():
    """Verify value_and_grad works identically with checkpointed function."""

    def f(x):
        return ops.multiply(x, ops.multiply(x, x))

    f_cp = checkpoint(f)

    x1 = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    x2 = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    vg_std = value_and_grad(f)
    vg_cp = value_and_grad(f_cp)

    v1, g1 = vg_std(x1)
    v2, g2 = vg_cp(x2)

    np.testing.assert_allclose(get_active_backend().asarray(v1), get_active_backend().asarray(v2))
    np.testing.assert_allclose(get_active_backend().asarray(g1), get_active_backend().asarray(g2))


def test_checkpoint_eager_execution():
    """Verify checkpointed function executes correctly in eager mode."""

    def f(x):
        return ops.add(x, 10.0)

    f_cp = checkpoint(f)

    t = Tensor(np.array([5.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        out = f_cp(t)

    assert float(get_active_backend().asarray(out)[0]) == 15.0
