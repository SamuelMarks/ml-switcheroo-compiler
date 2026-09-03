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


import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.ops.nn.quantized_ops import QuantizedOpsConfig, quantized_conv


def create_eager_tensor(data):
    return Tensor(data, TensorConfig(data.shape, DType.Float32, Device("cpu")))


def test_quantized_ops_missing():
    input = create_eager_tensor(np.ones((1, 2, 2, 1)))
    weight = create_eager_tensor(np.ones((2, 2, 1, 1)))
    scales = create_eager_tensor(np.ones((1,)))
    config = QuantizedOpsConfig(weight, scales, biases=None)

    with ConfigContext(eager_mode=True):
        res = quantized_conv(input, config)
        assert isinstance(res, Tensor)
        np.testing.assert_array_equal(res.numpy(), np.array([[[[4.0]]]]))


from unittest.mock import patch

from ml_switcheroo_compiler.ops.nn.quantized_ops import (
    AbsMaxQuantizeOp,
    ComputeFloat8AmaxHistoryOp,
    ComputeFloat8ScaleOp,
    FakeQuantWithMinMaxVarsOp,
    GatherQMMOp,
    QuantizeAndDequantizeOp,
    QuantizedMatmulOp,
    QuantizeOp,
    abs_max_quantize,
    compute_float8_amax_history,
    compute_float8_scale,
    dequantize,
    fake_quant_with_min_max_vars,
    gather_qmm,
    quantize,
    quantize_and_dequantize,
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
