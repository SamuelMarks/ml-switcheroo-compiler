import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention import ScaledDotProductAttention, scaled_dot_product_attention
from ml_switcheroo_compiler.tracing.tracer import TracerTape


def test_scaled_dot_product_attention_eager():
    q = Tensor(np.array([[[1.0]]]), TensorConfig((1, 1, 1), "float32", "cpu"))
    k = Tensor(np.array([[[1.0]]]), TensorConfig((1, 1, 1), "float32", "cpu"))
    v = Tensor(np.array([[[2.0]]]), TensorConfig((1, 1, 1), "float32", "cpu"))
    scale = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))

    config.eager_mode = True
    out = scaled_dot_product_attention(q, k, v, scale)
    assert isinstance(out, Tensor)


def test_scaled_dot_product_attention_trace():
    q = Tensor(np.array([[[1.0]]]), TensorConfig((1, 1, 1), "float32", "cpu"))
    k = Tensor(np.array([[[1.0]]]), TensorConfig((1, 1, 1), "float32", "cpu"))
    v = Tensor(np.array([[[2.0]]]), TensorConfig((1, 1, 1), "float32", "cpu"))
    scale = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))

    config.eager_mode = False

    tracer = TracerTape()
    tracer.start_tracing()
    try:
        out = scaled_dot_product_attention(q, k, v, scale)
        assert out is not None
    finally:
        tracer.stop_tracing()


def test_scaled_dot_product_attention_opdef():
    op = ScaledDotProductAttention()

    class DummyShape:
        shape = (1, 1, 1)

    res = op.infer_shape(DummyShape(), None, None, None)
    assert res == (1, 1, 1)
