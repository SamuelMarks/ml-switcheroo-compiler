from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_pool import UnpoolOptions, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool2d, adaptive_max_pool3d, fold, fractional_max_pool2d, fractional_max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d, unfold
from ml_switcheroo_compiler.ops.reductions.frontend_utils import WindowConfig, _emit_reduction_node, reduce_window


def test_reductions_frontend_brute():
    config.backend = "numpy"
    config.eager_mode = True

    t_2d = Tensor(np.random.rand(2, 3, 8, 8).astype(np.float32), TensorConfig((2, 3, 8, 8), "float32", "cpu"))
    t_3d = Tensor(np.random.rand(2, 3, 8, 8, 8).astype(np.float32), TensorConfig((2, 3, 8, 8, 8), "float32", "cpu"))
    t_1d = Tensor(np.random.rand(2, 3, 8).astype(np.float32), TensorConfig((2, 3, 8), "float32", "cpu"))
    indices = Tensor(np.zeros((2, 3, 8)).astype(np.int64), TensorConfig((2, 3, 8), "int64", "cpu"))
    indices2d = Tensor(np.zeros((2, 3, 8, 8)).astype(np.int64), TensorConfig((2, 3, 8, 8), "int64", "cpu"))
    indices3d = Tensor(np.zeros((2, 3, 8, 8, 8)).astype(np.int64), TensorConfig((2, 3, 8, 8, 8), "int64", "cpu"))

    with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool.get_active_backend") as mock_backend:

        class DummyBackend:
            def execute_op(self, op_type, *args, **kwargs):
                if return_indices := kwargs.get("return_indices", False) or op_type == "FractionalMaxPool3d":
                    return np.zeros_like(args[0]), np.zeros_like(args[0]).astype(np.int64)
                return np.zeros_like(args[0])

            def array(self, x):
                return x

        mock_backend.return_value = DummyBackend()

        adaptive_avg_pool2d(t_2d, (4, 4))
        adaptive_max_pool2d(t_2d, (4, 4))

        fractional_max_pool3d(t_3d, (4, 4, 4))
        adaptive_avg_pool3d(t_3d, (4, 4, 4))
        adaptive_max_pool3d(t_3d, (4, 4, 4))
        adaptive_max_pool3d(t_3d, (4, 4, 4), return_indices=True)

        opt1 = UnpoolOptions(2, output_size=(16,))
        opt2 = UnpoolOptions((2, 2), output_size=(16, 16))
        opt3 = UnpoolOptions((2, 2, 2), output_size=(16, 16, 16))
        max_unpool1d(t_1d, indices, opt1)
        max_unpool2d(t_2d, indices2d, opt2)
        max_unpool3d(t_3d, indices3d, opt3)

        w_conf = WindowConfig((1, 1, 2), (1, 1, 2), ((0, 0), (0, 0), (0, 0)), (1, 1, 1), (1, 1, 1))
        reduce_window(t_1d, 0.0, "max", w_conf)
        t_scalar = Tensor(np.array(0.0).astype(np.float32), TensorConfig((), "float32", "cpu"))
        reduce_window(t_1d, t_scalar, "max", w_conf)

    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    try:
        with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool._emit_reduction_node") as mock_emit, patch("ml_switcheroo_compiler.ops.reductions.frontend_utils._emit_reduction_node") as mock_emit_u:
            mock_emit.return_value = t_2d
            mock_emit_u.return_value = t_1d

            fractional_max_pool2d(t_2d, (4, 4))
            adaptive_avg_pool2d(t_2d, (4, 4))
            adaptive_max_pool2d(t_2d, (4, 4))
            unfold(t_2d, (2, 2))
            fold(t_2d, (8, 8), (2, 2))

            fractional_max_pool3d(t_3d, (4, 4, 4))
            adaptive_avg_pool3d(t_3d, (4, 4, 4))
            adaptive_max_pool3d(t_3d, (4, 4, 4))
            adaptive_max_pool3d(t_3d, (4, 4, 4), return_indices=True)

            max_unpool1d(t_1d, indices, opt1)
            max_unpool2d(t_2d, indices2d, opt2)
            max_unpool3d(t_3d, indices3d, opt3)

            reduce_window(t_1d, 0.0, "max", w_conf)
            reduce_window(t_1d, t_1d, "max", w_conf)
    finally:
        global_tracing_state.is_tracing = False

    global_tracing_state.is_tracing = True
    try:

        class DummyData:
            id = "dummy"

        t_dummy = Tensor(DummyData(), TensorConfig((2,), "float32", "cpu"))
        with patch("ml_switcheroo_compiler.ops.reductions.frontend_utils.global_tracing_state.add_node") as mock_add_node:
            _emit_reduction_node("Test", [t_dummy], {}, (2,), "float32")
    finally:
        global_tracing_state.is_tracing = False
