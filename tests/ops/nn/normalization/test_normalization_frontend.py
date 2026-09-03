from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.normalization.frontend import NormConfig, group_mean, group_norm, group_variance, spectral_normalization


def test_normalization_frontend_brute():
    config.backend = "numpy"
    config.eager_mode = True

    t_in = Tensor(np.random.rand(2, 4, 8, 8).astype(np.float32), TensorConfig((2, 4, 8, 8), "float32", "cpu"))
    u_in = Tensor(np.random.rand(4).astype(np.float32), TensorConfig((4,), "float32", "cpu"))

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:

        class DummyBackend:
            def execute_op(self, *args, **kwargs):
                return np.zeros((2, 4, 8, 8)).astype(np.float32)

            def array(self, x):
                return x

        mock_backend.return_value = DummyBackend()

        group_mean(t_in, 2)
        group_variance(t_in, 2)
        group_norm(t_in, 2)
        group_norm(t_in, 2, config=NormConfig(weight=t_in, bias=t_in))

        import sys

        norm_frontend = sys.modules["ml_switcheroo_compiler.ops.normalization.frontend"]
        with patch.object(norm_frontend, "power_iteration") as mock_pi:
            mock_pi.return_value = (t_in, u_in, t_in)
            spectral_normalization(t_in, u_in)

    # Tracing
    config.eager_mode = False
    with patch.object(norm_frontend, "get_op") as mock_get_op:

        def dummy_op(*args, **kwargs):
            return t_in

        mock_get_op.return_value = lambda: dummy_op

        group_mean(t_in, 2)
        group_variance(t_in, 2)
        group_norm(t_in, 2)
        group_norm(t_in, 2, config=NormConfig())

        norm_frontend = sys.modules["ml_switcheroo_compiler.ops.normalization.frontend"]
        with patch.object(norm_frontend, "power_iteration") as mock_pi:
            mock_pi.return_value = (t_in, u_in, t_in)
            with patch.object(norm_frontend, "divide") as mock_div:
                mock_div.return_value = t_in
                spectral_normalization(t_in, u_in)
