from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_frontend_pool_coverage():
    from unittest.mock import MagicMock, patch

    import numpy as np

    import ml_switcheroo_compiler.ops.reductions.frontend_pool as pool

    orig = config.eager_mode
    config.eager_mode = False

    t = Tensor(np.ones((1, 1, 4, 4)), TensorConfig(shape=(1, 1, 4, 4), dtype=DType("float32"), device=Device("cpu")))
    t_3d = Tensor(np.ones((1, 1, 4, 4, 4)), TensorConfig(shape=(1, 1, 4, 4, 4), dtype=DType("float32"), device=Device("cpu")))

    try:
        with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool._emit_reduction_node", return_value=t):
            pool.fractional_max_pool2d(t, (2, 2))
            pool.adaptive_avg_pool2d(t, (2, 2))
            pool.adaptive_max_pool2d(t, (2, 2))
            pool.unfold(t, (2, 2))
            pool.fold(t, (2, 2), (2, 2))

            pool.fractional_max_pool3d(t_3d, (2, 2, 2))
            pool.adaptive_avg_pool3d(t_3d, (2, 2, 2))
            pool.adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=False)
            pool.adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=True)

            opt = pool.UnpoolOptions(kernel_size=(2, 2), output_size=(4, 4))
            opt1d = pool.UnpoolOptions(kernel_size=(2,), output_size=(4,))
            opt3d = pool.UnpoolOptions(kernel_size=(2, 2, 2), output_size=(4, 4, 4))
            pool.max_unpool1d(t, t, opt1d)
            pool.max_unpool2d(t, t, opt)
            pool.max_unpool3d(t_3d, t_3d, opt3d)

            # None output_size
            opt_none = pool.UnpoolOptions(kernel_size=(2, 2), output_size=None)
            pool.max_unpool1d(t, t, opt_none)
            pool.max_unpool2d(t, t, opt_none)
            pool.max_unpool3d(t_3d, t_3d, opt_none)

        config.eager_mode = True

        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = np.zeros((1, 1, 2, 2))
        mock_backend.array = lambda x: x

        mock_backend_3d = MagicMock()
        mock_backend_3d.execute_op.return_value = (np.zeros((1, 1, 2, 2, 2)), np.zeros((1, 1, 2, 2, 2)))
        mock_backend_3d.array = lambda x: x

        with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool.get_active_backend", return_value=mock_backend):
            pool.adaptive_avg_pool2d(t, (2, 2))
            pool.adaptive_max_pool2d(t, (2, 2))
            pool.max_unpool1d(t, t, opt1d)
            pool.max_unpool2d(t, t, opt)
            pool.max_unpool3d(t_3d, t_3d, opt3d)

            mock_backend.execute_op.return_value = np.zeros((1, 1, 2, 2, 2))
            pool.adaptive_avg_pool3d(t_3d, (2, 2, 2))
            pool.adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=False)

        with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool.get_active_backend", return_value=mock_backend_3d):
            pool.fractional_max_pool3d(t_3d, (2, 2, 2), random_samples=t_3d)
            pool.adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=True)

            # and test FractionalMaxPool3d eager mode with no random samples
            pool.fractional_max_pool3d(t_3d, (2, 2, 2))

    finally:
        config.eager_mode = orig
