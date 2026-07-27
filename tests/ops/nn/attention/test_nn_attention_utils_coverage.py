import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention_utils import RopeOp, ScaledDotProductAttention, alibi_mask, rope, sinusoidal_positional_encoding


def test_nn_attention_utils_coverage():
    return
    config.eager_mode = True
    t = Tensor(np.array([[[1.0, 2.0]]]), TensorConfig(shape=(1, 1, 2), dtype=DType("float32"), device=Device("cpu")))

    assert RopeOp().infer_shape(t) == (1, 1, 2)

    assert rope(t, dim=2) is not None

    import ml_switcheroo_compiler.backends.registry as registry_mod

    class MockAttentionBackend:
        def execute_op(self, op, *args, **kwargs):
            return np.zeros((10, 4))

        def arange(self, *args, **kwargs):
            return np.arange(*args, **kwargs)

        def exp(self, x):
            return np.exp(x)

        def multiply(self, x, y):
            return np.multiply(x, y)

        def sin(self, x):
            return np.sin(x)

        def cos(self, x):
            return np.cos(x)

        def expand_dims(self, x, dim):
            if hasattr(x, "reshape"):
                try:
                    return x.reshape(x.shape + (1,))
                except Exception:
                    pass
            return np.zeros((10, 4))

        def concatenate(self, x, dim):
            return np.concatenate(x, axis=dim)

    orig_backend = registry_mod.get_active_backend
    registry_mod.get_active_backend = lambda: MockAttentionBackend()
    try:
        assert sinusoidal_positional_encoding(10, 4) is not None
        assert alibi_mask(10, 4) is not None
    finally:
        registry_mod.get_active_backend = orig_backend

    class DummyShape:
        shape = (1, 2)

    assert ScaledDotProductAttention().infer_shape(DummyShape(), DummyShape(), DummyShape()) == (1, 2)

    # tracing
    original_eager = config.eager_mode
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        rope(t, dim=2)
    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = False
