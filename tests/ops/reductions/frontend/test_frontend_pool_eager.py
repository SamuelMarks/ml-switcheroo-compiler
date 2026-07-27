from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_pool import UnpoolOptions, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool2d, adaptive_max_pool3d, fractional_max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d


def test_frontend_pool_eager():
    class MockBackend:
        def execute_op(self, *args, **kwargs):
            if args[0] in ("FractionalMaxPool3D", "AdaptiveMaxPool3D"):
                if kwargs.get("return_indices", True):
                    return np.array([1.0]), np.array([1])
            if args[0] == "FractionalMaxPool2D" or args[0] == "AdaptiveMaxPool2D":
                return np.array([1.0])
            return np.array([1.0])

        def array(self, x):
            return np.array(x)

    orig = config.eager_mode
    config.eager_mode = True
    try:
        with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool.get_active_backend", return_value=MockBackend()):
            t3 = Tensor(np.array([[[1.0]]]), TensorConfig(shape=(1, 1, 1), dtype=DType("float32"), device=Device("cpu")))
            t4 = Tensor(np.array([[[[1.0]]]]), TensorConfig(shape=(1, 1, 1, 1), dtype=DType("float32"), device=Device("cpu")))
            t5 = Tensor(np.array([[[[[1.0]]]]]), TensorConfig(shape=(1, 1, 1, 1, 1), dtype=DType("float32"), device=Device("cpu")))

            adaptive_avg_pool2d(t4, output_size=[1, 1])
            adaptive_max_pool2d(t4, output_size=[1, 1])
            fractional_max_pool3d(t5, output_size=[1, 1, 1])
            adaptive_avg_pool3d(t5, output_size=[1, 1, 1])
            adaptive_max_pool3d(t5, output_size=[1, 1, 1], return_indices=True)
            adaptive_max_pool3d(t5, output_size=[1, 1, 1], return_indices=False)
            max_unpool1d(t3, t3, UnpoolOptions(kernel_size=(1,), output_size=(1,)))
            max_unpool2d(t4, t4, UnpoolOptions(kernel_size=(1, 1), output_size=(1, 1)))
            max_unpool3d(t5, t5, UnpoolOptions(kernel_size=(1, 1, 1), output_size=(1, 1, 1)))
    finally:
        config.eager_mode = orig
