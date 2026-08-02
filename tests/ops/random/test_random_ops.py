# ruff: noqa: E501
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.random_ops import (
    Ball,
    Beta,
    Binomial,
    Bits,
    Cauchy,
    Chisquare,
    Clone,
    Dirichlet,
    DoubleSidedMaxwell,
    Exponential,
    F,
    Gamma,
    GeneralizedNormal,
    Gumbel,
    Key,
    KeyData,
    KeyImpl,
    Laplace,
    Loggamma,
    Logistic,
    Lognormal,
    Maxwell,
    MultivariateNormal,
    Orthogonal,
    Pareto,
    Poisson,
    RandomGammaP,
    Rayleigh,
    RngBitGenerator,
    RngUniform,
    T,
    Triangular,
    Wald,
    WeibullMin,
    WrapKeyData,
)
from ml_switcheroo_compiler.ops.random_ops.frontend import sobol_sample
from ml_switcheroo_compiler.ops.random_ops.sobol import SobolSample, generate_sobol


def test_random_ops_coverage():
    config.eager_mode = True

    class DummyShape:
        shape = (1, 2)

    ops = [
        Binomial,
        Cauchy,
        Chisquare,
        Dirichlet,
        DoubleSidedMaxwell,
        Exponential,
        F,
        Gumbel,
        Laplace,
        Loggamma,
        Logistic,
        Lognormal,
        Maxwell,
        MultivariateNormal,
        Pareto,
        Poisson,
        Rayleigh,
        T,
        Triangular,
        Wald,
        WeibullMin,
        Clone,
        KeyData,
        KeyImpl,
        WrapKeyData,
        Bits,
        GeneralizedNormal,
        Orthogonal,
        RandomGammaP,
        Ball,
        Key,
        Beta,
        Gamma,
    ]

    for op_cls in ops:
        op = op_cls()
        assert op.infer_shape(DummyShape()) is not None
        assert op.infer_shape() == ()

    assert RngBitGenerator().infer_shape(1, (1, 2), "float32") == (1, 2)
    assert RngUniform().infer_shape(0.0, 1.0, (1, 2), "float32") == (1, 2)
    assert SobolSample().infer_shape(2, 10, skip=0) == (10, 2)

    assert generate_sobol(2, 10, 0).shape == (10, 2)

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((10, 2))
        assert sobol_sample(2, 10, 0) is not None

        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        assert sobol_sample(2, 10, 0) is not None

        global_tracing_state.is_tracing = False
        with pytest.raises(RuntimeError):
            sobol_sample(2, 10, 0)

        config.eager_mode = True
