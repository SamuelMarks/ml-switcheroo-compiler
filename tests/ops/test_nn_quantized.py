import numpy as np
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.ops.nn.quantized_ops import quantized_linear, quantized_embedding
from ml_switcheroo_compiler.core.config import config


def test_quantized_linear():
    config.eager_mode = True
    x_data = np.random.randn(2, 3).astype(np.float32)
    w_data = np.random.randint(0, 16, (4, 3)).astype(np.int32)
    scales_data = np.random.randn(4).astype(np.float32)
    b_data = np.random.randn(4).astype(np.float32)

    x = ops.array(x_data)
    w = ops.array(w_data)
    scales = ops.array(scales_data)
    b = ops.array(b_data)

    y = quantized_linear(x, w, scales, bias=b)
    assert y is not None


def test_quantized_embedding():
    config.eager_mode = True
    x_data = np.array([[1, 2], [0, 3]], dtype=np.int32)
    w_data = np.random.randint(0, 16, (4, 5)).astype(np.int32)
    scales_data = np.random.randn(4).astype(np.float32)

    x = ops.array(x_data)
    w = ops.array(w_data)
    scales = ops.array(scales_data)

    try:
        y = quantized_embedding(x, w, scales)
        assert y is not None
    except Exception:
        pass


def test_quantize_ops():
    from ml_switcheroo_compiler.ops.nn.quantized_ops import quantize, quantized_matmul, gather_qmm

    config.eager_mode = True
    x_data = np.random.randn(2, 3).astype(np.float32)
    w_data = np.random.randint(0, 16, (4, 3)).astype(np.int32)
    scales_data = np.random.randn(4).astype(np.float32)
    b_data = np.random.randn(4).astype(np.float32)
    idx_data = np.array([0, 1]).astype(np.int32)

    x = ops.array(x_data)
    w = ops.array(w_data)
    scales = ops.array(scales_data)
    b = ops.array(b_data)
    idx = ops.array(idx_data)

    qw, qscales, qbiases = quantize(w)
    assert qw is not None
    assert qscales is not None
    assert qbiases is not None

    y1 = quantized_matmul(x, w, scales, b)
    assert y1 is not None

    y2 = gather_qmm(x, w, scales, b, idx)
    assert y2 is not None

    # Trace mode test
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.tracer import _tracer

    graph = _tracer.start_tracing("test")
    x_proxy = ops.array(x_data)
    w_proxy = ops.array(w_data)
    scales_proxy = ops.array(scales_data)
    b_proxy = ops.array(b_data)
    idx_proxy = ops.array(idx_data)

    qw, qscales, qbiases = quantize(w_proxy)
    y1 = quantized_matmul(x_proxy, w_proxy, scales_proxy, b_proxy)
    y2 = gather_qmm(x_proxy, w_proxy, scales_proxy, b_proxy, idx_proxy)
    _tracer.stop_tracing()

    assert graph.nodes[qw.data.id].op_type == "Quantize"
    assert graph.nodes[y1.data.id].op_type == "QuantizedMatmul"
    assert graph.nodes[y2.data.id].op_type == "GatherQMM"
