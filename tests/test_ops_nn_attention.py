import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention import ScaledDotProductAttention, scaled_dot_product_attention
from ml_switcheroo_compiler.tracing.tracer import TracerTape


def test_scaled_dot_product_attention_eager_2():
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


def test_scaled_dot_product_attention_eager():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.nn.attention import scaled_dot_product_attention

    config.eager_mode = True
    try:
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            backend = MagicMock()
            backend.execute_op.return_value = MagicMock(shape=(2, 2))
            mock_backend.return_value = backend

            import numpy as np

            q = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
            k = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
            v = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
            sf = Tensor(np.array(1.0), TensorConfig((), "float32", "cpu"))

            res = scaled_dot_product_attention(q, k, v, sf)
            assert isinstance(res, Tensor)

    finally:
        config.eager_mode = False


def test_scaled_dot_product_attention_infer_shape():
    from ml_switcheroo_compiler.ops.registry import get_op

    opCls = get_op("ScaledDotProductAttention")
    op = opCls()

    class Dummy:
        shape = (4, 4)

    assert op.infer_shape(q=Dummy(), k=None, v=None, scale_factor=None) == (4, 4)
