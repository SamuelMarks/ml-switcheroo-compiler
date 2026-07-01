from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.linalg.frontend import block_masked_mm
from ml_switcheroo_compiler.tracing import _tracer


def test_block_masked_mm_shape():
    config.eager_mode = False
    config.default_device = None

    _tracer.start_tracing()
    try:
        a = Tensor(None, TensorConfig((2, 64, 128), DType.Float32, None))
        b = Tensor(None, TensorConfig((2, 128, 64), DType.Float32, None))

        out = block_masked_mm(a, b, block_size=64)
        assert out.shape == (2, 64, 64)
        assert out.dtype == DType.Float32
    finally:
        _tracer.stop_tracing()


def test_block_masked_mm_eager():
    config.eager_mode = True
    config.default_device = None
    config.backend = "numpy"

    import numpy as np

    a_data = np.random.randn(2, 64, 128).astype(np.float32)
    b_data = np.random.randn(2, 128, 64).astype(np.float32)
    a = Tensor(a_data, TensorConfig((2, 64, 128), DType.Float32, None))
    b = Tensor(b_data, TensorConfig((2, 128, 64), DType.Float32, None))

    out = block_masked_mm(a, b, block_size=64)
    assert out.shape == (2, 64, 64)
