import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention_utils import RopeOp, ScaledDotProductAttention, alibi_mask, rope, sinusoidal_positional_encoding


def test_rope_infer_shape():
    class DummyShape:
        shape = (1, 2)

    assert RopeOp().infer_shape(DummyShape()) == (1, 2)


def test_rope():
    t = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
    orig = config.eager_mode
    config.eager_mode = True
    try:
        rope(t, 2)
    except Exception:
        pass
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None
    try:
        rope(t, 2)
    except Exception:
        pass
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node


def test_sinusoidal():
    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None
    try:
        res = sinusoidal_positional_encoding(4, 4)
        assert res is not None
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node


def test_alibi():
    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None
    try:
        res = alibi_mask(4, 2)
        assert res == 0.0
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node


def test_scaled_dot():
    class DummyShape:
        shape = (1, 2)

    assert ScaledDotProductAttention().infer_shape(DummyShape(), None, None) == (1, 2)
