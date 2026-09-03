import numpy as np
import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.attention import ScaledDotProductAttention, scaled_dot_product_attention


def test_dot_product_attention_eager_branch():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    q = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    k = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    v = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    sf = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))

    from unittest.mock import MagicMock

    mock_b = MagicMock()
    mock_b.execute_op.return_value = np.ones((2, 2))

    with pytest.MonkeyPatch.context() as m:
        m.setattr("ml_switcheroo_compiler.ops.nn.attention.get_active_backend", lambda: mock_b)
        res = scaled_dot_product_attention(q, k, v, sf)
        assert res.shape == (2, 2)

    config.eager_mode = False


def test_dot_product_attention_trace_branch():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    q = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    k = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    v = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    sf = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))

    res = scaled_dot_product_attention(q, k, v, sf)
    assert res.shape == (2, 2)


def test_dot_product_attention_infer_shape():
    op = ScaledDotProductAttention()
    q = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    k = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    v = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    sf = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    assert op.infer_shape(q, k, v, sf) == (2, 2)


def test_quantized_trace():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    from ml_switcheroo_compiler.ops.nn.quantized_ops import fake_quantize_per_channel_affine, fake_quantize_per_tensor_affine

    q = Tensor(np.ones((2, 2)), TensorConfig((2, 2), DType.Float32, Device("cpu")))
    sf = Tensor(np.ones((2,)), TensorConfig((2,), DType.Float32, Device("cpu")))
    zp = Tensor(np.ones((2,)), TensorConfig((2,), DType.Float32, Device("cpu")))

    res = fake_quantize_per_channel_affine(q, sf, zp, axis=0, quant_min=0, quant_max=255)
    assert res is not None

    sf2 = Tensor(np.ones((1,)), TensorConfig((1,), DType.Float32, Device("cpu")))
    zp2 = Tensor(np.ones((1,)), TensorConfig((1,), DType.Float32, Device("cpu")))
    res2 = fake_quantize_per_tensor_affine(q, sf2, zp2, quant_min=0, quant_max=255)
    assert res2 is not None
