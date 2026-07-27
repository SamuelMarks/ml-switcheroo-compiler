from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_quantized_ops_eager():
    import numpy as np

    import ml_switcheroo_compiler.ops.nn.quantized_ops as q_ops

    orig = config.eager_mode
    config.eager_mode = True

    try:
        t = Tensor(np.array([[1.0, 2.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))

        q_ops.fake_quant_with_min_max_vars(t, t, t)

        qp = q_ops.QuantizationParams()
        q_ops.quantize_and_dequantize(t, t, t, qp)

        q_ops.abs_max_quantize(t)

        q_ops.compute_float8_amax_history(t, t)
        q_ops.compute_float8_scale(t, t)

        q_ops.quantize(t)

        qc = q_ops.QuantizationConfig()
        qoc = q_ops.QuantizedOpsConfig(weight=t, scales=t, biases=t, indices=t, q_config=qc)

        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.ops.nn.linear_ops.linear", return_value=t):
            q_ops.quantized_matmul(t, qoc)
            q_ops.gather_qmm(t, qoc)

        # Test actual eager dequantization mathematical accuracy
        w_quant = Tensor(np.array([[2.0, 4.0]]), TensorConfig((1, 2), "float32", "cpu"))
        scales = Tensor(np.array([[0.5, 0.5]]), TensorConfig((1, 2), "float32", "cpu"))
        zeros = Tensor(np.array([[1.0, 1.0]]), TensorConfig((1, 2), "float32", "cpu"))

        # W_float = (W_quant - zeros) * scales = ([[2, 4]] - [[1, 1]]) * [[0.5, 0.5]] = [[1, 3]] * [[0.5, 0.5]] = [[0.5, 1.5]]
        qoc_math = q_ops.QuantizedOpsConfig(weight=w_quant, scales=scales, zeros=zeros, biases=None, indices=None, q_config=None)

        # Test quantized_embedding math
        indices = Tensor(np.array([[0]]), TensorConfig((1, 1), "int32", "cpu"))
        embed_res = q_ops.quantized_embedding(indices, qoc_math)
        assert np.allclose(embed_res.data, [[0.5, 1.5]])

        # Test quantized_linear math (input of ones)
        inp = Tensor(np.array([[1.0, 1.0]]), TensorConfig((1, 2), "float32", "cpu"))
        # we expect input @ weight_float.T = [[1.0, 1.0]] @ [[0.5, 1.5]].T = 0.5 * 1.0 + 1.5 * 1.0 = 2.0
        lin_res = q_ops.quantized_linear(inp, qoc_math)
        assert np.allclose(lin_res.data, [[2.0]])

        # Test dequantize mathematical correctness
        deq_res = q_ops.dequantize(w_quant, scales, zeros)
        assert np.allclose(deq_res.data, [[0.5, 1.5]])

        # Coverage for zeros=None, biases=None in eager mode
        qoc_none = q_ops.QuantizedOpsConfig(weight=w_quant, scales=scales, zeros=None, biases=None, indices=None, q_config=None)
        embed_none = q_ops.quantized_embedding(indices, qoc_none)
        assert embed_none is not None
        lin_none = q_ops.quantized_linear(inp, qoc_none)
        assert lin_none is not None
        deq_none = q_ops.dequantize(w_quant, scales, biases=None)
        assert deq_none is not None

        # Test quantized_conv 2D mathematical correctness
        input_2d_val = np.ones((1, 3, 3, 1), dtype=np.float32)
        input_2d = Tensor(input_2d_val, TensorConfig((1, 3, 3, 1), DType.Float32, Device("cpu")))

        w_2d_quant = Tensor(np.array([[[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]]]), TensorConfig((2, 2, 1, 1), DType.Float32, Device("cpu")))
        scales_2d = Tensor(np.ones((2, 2, 1, 1), dtype=np.float32) * 0.5, TensorConfig((2, 2, 1, 1), DType.Float32, Device("cpu")))
        zeros_2d = Tensor(np.ones((2, 2, 1, 1), dtype=np.float32) * 1.0, TensorConfig((2, 2, 1, 1), DType.Float32, Device("cpu")))

        # dequantized weight will be ones * 0.5
        # Convolution of a 3x3 input of ones with 2x2 kernel of 0.5s results in a 2x2 output of:
        # 0.5 * 1.0 * 4 = 2.0 per output patch (under VALID padding)
        qoc_conv = q_ops.QuantizedOpsConfig(weight=w_2d_quant, scales=scales_2d, biases=zeros_2d, indices=None, q_config=None)

        conv_res = q_ops.quantized_conv(input_2d, qoc_conv, stride=1, padding="VALID")
        assert np.allclose(conv_res.data, [[[[2.0]], [[2.0]]], [[[2.0]], [[2.0]]]])

    finally:
        config.eager_mode = orig
