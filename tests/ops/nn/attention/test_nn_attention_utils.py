import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention_utils import RopeOp, rope
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_nn_attention_utils_coverage():
    config.eager_mode = True
    t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig(shape=(1, 1, 2), dtype=DType("float32"), device=Device("cpu")))

    assert RopeOp().infer_shape(t) == (1, 1, 2)

    assert rope(t, axis=2) is not None

    import ml_switcheroo_compiler.backends.registry as registry_mod

    class MockAttentionBackend:
        def execute_op(self, op_name, *args, **kwargs):
            return "exec"

        def asarray(self, x):
            return np.asarray(x)

        def array(self, x):
            return np.asarray(x)

        def expand_dims(self, x, axis):
            return np.expand_dims(x, axis=axis)

        def multiply(self, x, y):
            try:
                return np.multiply(x, y)
            except Exception:
                return x

        def arange(self, *args, **kwargs):
            return np.arange(*args, **kwargs)

        def exp(self, x):
            return np.exp(x)

        def sin(self, x):
            return np.sin(x)

        def cos(self, x):
            return np.cos(x)

        def concatenate(self, x, axis):
            try:
                return np.concatenate(x, axis=axis)
            except Exception:
                return x

    orig_backend = registry_mod.get_active_backend
    try:
        registry_mod.get_active_backend = lambda: MockAttentionBackend()

        rope(t, axis=2)

        config.eager_mode = False
        global_tracing_state.is_tracing = True

        class DummyGraph:
            def __init__(self):
                self.name = "dummy"
                self.nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        rope(t, axis=2)
    finally:
        config.eager_mode = False
        global_tracing_state.is_tracing = False
        registry_mod.get_active_backend = orig_backend
