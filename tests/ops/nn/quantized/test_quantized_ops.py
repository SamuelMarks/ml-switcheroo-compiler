from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_quantized_ops_trace():
    from unittest.mock import patch

    import numpy as np

    import ml_switcheroo_compiler.ops.nn.quantized_ops as q_ops

    orig = config.eager_mode
    config.eager_mode = False

    try:
        t = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))

        with patch("ml_switcheroo_compiler.ops.nn.quantized_ops._emit_shape_node", return_value=t), patch("ml_switcheroo_compiler.ops.nn.quantized_ops.linear", return_value=t), patch("ml_switcheroo_compiler.ops.nn.quantized_ops.gather", return_value=t):
            q_ops.fake_quant_with_min_max_vars(t, t, t)

            qp = q_ops.QuantizationParams()
            q_ops.quantize_and_dequantize(t, t, t, qp)

            q_ops.abs_max_quantize(t)

            q_ops.compute_float8_amax_history(t, t)
            q_ops.compute_float8_scale(t, t)

            q_ops.quantize(t)

            qc = q_ops.QuantizationConfig()
            qoc = q_ops.QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=qc)

            q_ops.quantized_matmul(t, qoc)
            q_ops.gather_qmm(t, qoc)

            # Test infer_shapes
            for op_class in [q_ops.QuantizeOp, q_ops.QuantizedMatmulOp, q_ops.GatherQMMOp, q_ops.FakeQuantWithMinMaxVarsOp, q_ops.QuantizeAndDequantizeOp, q_ops.AbsMaxQuantizeOp, q_ops.ComputeFloat8AmaxHistoryOp, q_ops.ComputeFloat8ScaleOp]:
                op = op_class()
                assert op.infer_shape(t) == (1, 2)

            q_ops.quantized_linear(t, qoc)
            q_ops.quantized_embedding(t, qoc)
            q_ops.dequantize(t, t, t)
            q_ops.quantized_conv(t, qoc)

            # test QuantizationParams combinations
            qoc_no_biases = q_ops.QuantizedOpsConfig(weight=t, scales=t, biases=None, indices=None, q_config=None)
            q_ops.quantized_matmul(t, qoc_no_biases)
            q_ops.gather_qmm(t, qoc_no_biases)
            q_ops.quantized_linear(t, qoc_no_biases)
            q_ops.dequantize(t, t, biases=None)
            q_ops.quantized_conv(t, qoc_no_biases)

            # Test params=None
            q_ops.quantize_and_dequantize(t, t, t, params=None)

    finally:
        config.eager_mode = orig
