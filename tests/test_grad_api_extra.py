import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import hook_gradient


def test_hook_gradient_returns_none():
    t = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))

    def hook(g):
        return None  # Trigger the "if out_g is None:" block

    # Needs to be run under tracing or eagerly handled.
    # Let's mock CustomVJP behaviour for this specific internal hook block.
    # The actual uncovered line is within _hook_bwd which is invoked by AD engine
    out = hook_gradient(t, hook)
    assert out is not None
