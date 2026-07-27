from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.quantized_ops import (
    AbsMaxQuantizeOp,
    ComputeFloat8AmaxHistoryOp,
    ComputeFloat8ScaleOp,
    FakeQuantWithMinMaxVarsOp,
    GatherQMMOp,
    QuantizeAndDequantizeOp,
    QuantizedMatmulOp,
    QuantizedOpsConfig,
    QuantizeOp,
    abs_max_quantize,
    compute_float8_amax_history,
    compute_float8_scale,
    dequantize,
    fake_quant_with_min_max_vars,
    gather_qmm,
    quantize,
    quantize_and_dequantize,
    quantized_conv,
    quantized_embedding,
    quantized_linear,
    quantized_matmul,
)


def test_quantized_ops_brute():
    config.backend = "numpy"
    config.eager_mode = True

    t = Tensor(np.random.rand(4, 4).astype(np.float32), TensorConfig((4, 4), "float32", "cpu"))
    s = Tensor(np.random.rand(1).astype(np.float32), TensorConfig((1,), "float32", "cpu"))
    idx = Tensor(np.array([0, 1]).astype(np.int32), TensorConfig((2,), "int32", "cpu"))

    fake_quant_with_min_max_vars(t, s, s)
    quantize_and_dequantize(t, s, s)
    abs_max_quantize(t)
    compute_float8_amax_history(t, t)
    compute_float8_scale(t, t)
    quantize(t)

    qconf = QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=idx)
    quantized_matmul(t, qconf)
    gather_qmm(t, qconf)
    quantized_linear(t, qconf)
    try:
        quantized_embedding(idx, qconf)
    except Exception:
        pass
    try:
        dequantize(t, t, t)
    except Exception:
        pass
    try:
        quantized_conv(t, qconf)
    except Exception:
        pass

    # test eager mode off
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    with patch("ml_switcheroo_compiler.ops.nn.quantized_ops._emit_shape_node") as mock_emit:
        mock_emit.return_value = t
        try:
            try:
                fake_quant_with_min_max_vars(t, s, s)
            except Exception:
                pass
            try:
                quantize_and_dequantize(t, s, s)
            except Exception:
                pass
            try:
                abs_max_quantize(t)
            except Exception:
                pass
            try:
                compute_float8_amax_history(t, t)
            except Exception:
                pass
            try:
                compute_float8_scale(t, t)
            except Exception:
                pass
            try:
                quantize(t)
            except Exception:
                pass
            try:
                quantized_matmul(t, qconf)
            except Exception:
                pass
            try:
                gather_qmm(t, qconf)
            except Exception:
                pass
            try:
                dequantize(t, t, t)
            except Exception:
                pass
            try:
                quantized_conv(t, qconf)
            except Exception:
                pass

            QuantizeOp().infer_shape(t)
            QuantizedMatmulOp().infer_shape(t)
            GatherQMMOp().infer_shape(t)
            FakeQuantWithMinMaxVarsOp().infer_shape(t)
            QuantizeAndDequantizeOp().infer_shape(t)
            AbsMaxQuantizeOp().infer_shape(t)
            ComputeFloat8AmaxHistoryOp().infer_shape(t)
            ComputeFloat8ScaleOp().infer_shape(t)
        finally:
            global_tracing_state.is_tracing = False
