from ml_switcheroo_compiler.ops.shape.indexing import take_along_axis
from ml_switcheroo_compiler.ops.signal import convolve2d, fftconvolve, welch
from ml_switcheroo_compiler.ops.stats import (
    norm_pdf,
    norm_cdf,
    gamma_pdf,
    gamma_cdf,
    beta_pdf,
    beta_cdf,
    poisson_pmf,
    poisson_cdf,
    binom_pmf,
    binom_cdf,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.ops.tensor import allclose
import numpy as np
from unittest.mock import patch


def test_misc():
    device = Device("cpu")
    t1 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            mock_backend.return_value.execute_op.return_value = np.ones((2,))
            convolve2d(t1, t1)
            fftconvolve(t1, t1)
            welch(t1)

            # just mock out eager eval for stats
            with patch(
                "ml_switcheroo_compiler.ops.eager_evaluator.EagerEvaluator.evaluate"
            ) as mock_eval:
                mock_eval.return_value = np.ones((2,))
                take_along_axis(t1, t1, 0)
                norm_pdf(t1)
                norm_cdf(t1)
                gamma_pdf(t1, t1)
                gamma_cdf(t1, t1)
                beta_pdf(t1, t1, t1)
                beta_cdf(t1, t1, t1)
                poisson_pmf(t1, t1)
                poisson_cdf(t1, t1)
                binom_pmf(t1, t1, t1)
                binom_cdf(t1, t1, t1)

                allclose(t1, t1)
