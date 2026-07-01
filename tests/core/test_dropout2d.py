from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.nn.dropout import dropout2d
from ml_switcheroo_compiler.tracing import _tracer


def test_dropout2d_shape():
    config.eager_mode = False
    config.default_device = None

    _tracer.start_tracing()
    try:
        a = Tensor(None, TensorConfig((2, 3, 64, 64), DType.Float32, None))

        out = dropout2d(a, p=0.5)
        assert out.shape == (2, 3, 64, 64)
        assert out.dtype == DType.Float32
    finally:
        _tracer.stop_tracing()


def test_dropout2d_eager():
    config.eager_mode = True
    config.default_device = None
    config.backend = "numpy"

    import numpy as np

    a_data = np.random.randn(2, 3, 64, 64).astype(np.float32)
    a = Tensor(a_data, TensorConfig((2, 3, 64, 64), DType.Float32, None))

    out = dropout2d(a, p=0.5)
    assert out.shape == (2, 3, 64, 64)
