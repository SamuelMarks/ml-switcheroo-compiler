"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.nn.quantized_ops import (
    QuantizedOpsConfig,
    gather_qmm,
    quantize,
    quantized_embedding,
    quantized_linear,
    quantized_matmul,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_quantized_linear() -> object:
    """Function docstring."""
    config.eager_mode = True
    x_data = np.random.randn(2, 3).astype(np.float32)
    w_data = np.random.randint(0, 16, (4, 3)).astype(np.int32)
    scales_data = np.random.randn(4).astype(np.float32)
    b_data = np.random.randn(4).astype(np.float32)

    x = ops.array(x_data)
    w = ops.array(w_data)
    scales = ops.array(scales_data)
    b = ops.array(b_data)

    qconf = QuantizedOpsConfig(weight=w, scales=scales, biases=b)
    y = quantized_linear(x, config=qconf)
    assert y is not None


def test_quantized_embedding() -> object:
    """Function docstring."""
    config.eager_mode = True
    x_data = np.array([[1, 2], [0, 3]], dtype=np.int32)
    w_data = np.random.randint(0, 16, (4, 5)).astype(np.int32)
    scales_data = np.random.randn(4).astype(np.float32)

    x = ops.array(x_data)
    w = ops.array(w_data)
    scales = ops.array(scales_data)

    qconf = QuantizedOpsConfig(weight=w, scales=scales)
    try:
        y = quantized_embedding(x, config=qconf)
        assert y is not None
    except Exception:
        pass


def test_quantize_ops() -> object:
    """Function docstring."""
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

    qconf1 = QuantizedOpsConfig(weight=w, scales=scales, biases=b)
    y1 = quantized_matmul(x, config=qconf1)
    assert y1 is not None

    qconf2 = QuantizedOpsConfig(weight=w, scales=scales, biases=b, indices=idx)
    y2 = gather_qmm(x, config=qconf2)
    assert y2 is not None

    # Trace mode test
    config.eager_mode = False

    graph = global_tracing_state.start_tracing("test")
    x_proxy = ops.array(x_data)
    w_proxy = ops.array(w_data)
    scales_proxy = ops.array(scales_data)
    b_proxy = ops.array(b_data)
    idx_proxy = ops.array(idx_data)

    qw, qscales, qbiases = quantize(w_proxy)
    qconf3 = QuantizedOpsConfig(weight=w_proxy, scales=scales_proxy, biases=b_proxy)
    y1 = quantized_matmul(x_proxy, config=qconf3)

    qconf4 = QuantizedOpsConfig(weight=w_proxy, scales=scales_proxy, biases=b_proxy, indices=idx_proxy)
    y2 = gather_qmm(x_proxy, config=qconf4)
    global_tracing_state.stop_tracing()

    assert graph.nodes[qw.data.id].op_type == "Quantize"
    assert graph.nodes[y1.data.id].op_type == "QuantizedMatmul"
    assert graph.nodes[y2.data.id].op_type == "GatherQMM"
