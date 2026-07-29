import numpy as np

import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import checkpoint, grad
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function


def test_checkpoint_equivalence():
    def f(x):
        y = ops.multiply(x, x)
        z = ops.multiply(y, x)
        return z

    f_cp = checkpoint(f)

    t_data = np.array(2.0, dtype=np.float32)
    t1 = Tensor(t_data, TensorConfig((), DType.Float32, Device("cpu")))
    t2 = Tensor(t_data.copy(), TensorConfig((), DType.Float32, Device("cpu")))

    g1 = grad(f)(t1)
    g2 = grad(f_cp)(t2)

    assert np.allclose(get_active_backend().asarray(g1), get_active_backend().asarray(g2))


def test_checkpoint_memory_usage():
    """Verify that checkpoint correctly reduces peak memory by checking the number of intermediate nodes in the forward graph."""

    def f(x):
        for _ in range(5):
            x = ops.multiply(x, x)
        return x

    f_cp = checkpoint(f)

    # Let's trace it and see
    t = Tensor(np.array(2.0, dtype=np.float32), TensorConfig((), DType.Float32, Device("cpu")))

    from ml_switcheroo_compiler.core.config import config

    prev_eager = config.eager_mode
    config.eager_mode = False

    # Trace f
    block_f = _trace_function(f, (t,), "f")
    # Trace f_cp
    block_f_cp = _trace_function(f_cp, (t,), "f_cp")

    config.eager_mode = prev_eager

    # In the f_cp block, there should only be a Checkpoint node and Output/Input nodes.
    # While block_f has many Multiply nodes.
    f_op_types = [n.op_type for n in block_f.nodes]
    f_cp_op_types = [n.op_type for n in block_f_cp.nodes]

    assert "Checkpoint" not in f_op_types
    assert "Checkpoint" in f_cp_op_types

    # f should have 5 multiplies
    assert f_op_types.count("Multiply") == 5
    # f_cp should have 0 multiplies in the outer block
    assert f_cp_op_types.count("Multiply") == 0


def test_checkpoint_eager_mode():
    def f(x):
        return ops.multiply(x, x)

    f_cp = checkpoint(f)
    t = Tensor(np.array(3.0, dtype=np.float32), TensorConfig((), DType.Float32, Device("cpu")))

    from ml_switcheroo_compiler.core.config import config

    prev_eager = config.eager_mode
    config.eager_mode = True
    out = f_cp(t)
    config.eager_mode = prev_eager

    assert np.allclose(get_active_backend().asarray(out), 9.0)
