import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
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
