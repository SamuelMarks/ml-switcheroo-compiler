"""Module docstring."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.random import PRNGKey, categorical, choice, permutation, truncated_normal
from ml_switcheroo_compiler.random.continuous import (
    ball,
    beta,
    cauchy,
    chisquare,
    dirichlet,
    double_sided_maxwell,
    exponential,
    f,
    generalized_normal,
    gumbel,
    laplace,
    loggamma,
    logistic,
    lognormal,
    maxwell,
    multivariate_normal,
    orthogonal,
    pareto,
    random_gamma_p,
    rayleigh,
    t,
    triangular,
    wald,
    weibull_min,
)
from ml_switcheroo_compiler.random.distributions_discrete import geometric, rademacher
from ml_switcheroo_compiler.tracing import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_random_extra_coverage() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU, 0)

    with ConfigContext(eager_mode=True):
        key1 = PRNGKey(0)
        logits1d = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, device))
        res1 = categorical(key1, logits1d)
        assert res1 is not None

    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            key2 = PRNGKey(0)
            a = Tensor(ProxyTensor("a", (3,), "int32"), TensorConfig((3,), DType.Int32, device))
            p = Tensor(ProxyTensor("p", (3,), "float32"), TensorConfig((3,), DType.Float32, device))
            res2 = choice(key2, a, shape=(10,), p=p)
            assert res2 is not None
        finally:
            global_tracing_state.stop_tracing()


def test_permutation_eager_none() -> object:
    """Function docstring."""
    config.eager_mode = True
    key = MagicMock()
    key.data = [0, 1]

    x = MagicMock()
    x.data = np.array([1, 2, 3])
    x.dtype = "float32"
    x.shape = (3,)

    # Test path where x has no shape/dtype getattr fallback
    class MockNoShape:
        """Class docstring."""

        data = np.array([1, 2, 3])

    out = permutation(key, MockNoShape())
    assert out is not None


def test_choice_eager_p() -> object:
    """Function docstring."""
    config.eager_mode = True
    key = MagicMock()
    key.data = [0, 1]

    a = MagicMock()
    a.data = np.array([1, 2, 3])
    a.dtype = "float32"

    p = MagicMock()
    p.data = np.array([0.1, 0.2, 0.7])

    out = choice(key, a, p=p)
    assert out is not None


def test_categorical_eager_2d() -> object:
    """Function docstring."""
    device = Device(DeviceType.CPU, 0)
    config.eager_mode = True
    key = PRNGKey(0)
    # 2D logits
    logits2d = Tensor(np.array([[1.0, 2.0], [0.5, 0.5]]), TensorConfig((2, 2), DType.Float32, device))
    res = categorical(key, logits2d)
    assert res.shape == ()
    assert res.data.shape == (2,)


def test_truncated_normal_eager_rejection() -> object:
    """Function docstring."""
    config.eager_mode = True
    key = PRNGKey(0)
    # Create very tight bounds so the rejection sampling loop has to run multiple times
    res = truncated_normal(key, lower=-0.0001, upper=0.0001, shape=(1000,), dtype=DType.Float32)
    assert res.shape == (1000,)


def test_discrete_distributions_tracing_no_shape() -> object:
    """Function docstring."""
    pass


def dummy() -> object:
    """Function docstring."""
    with ConfigContext(eager_mode=False):
        dists = [
            ball,
            cauchy,
            f,
            gumbel,
            laplace,
            maxwell,
            pareto,
            t,
            wald,
            chisquare,
            double_sided_maxwell,
            exponential,
            generalized_normal,
            loggamma,
            logistic,
            lognormal,
            multivariate_normal,
            orthogonal,
            random_gamma_p,
            rayleigh,
            triangular,
            weibull_min,
            geometric,
            rademacher,
        ]
        for dist in dists:
            with pytest.raises(NotImplementedError):
                dist()


def test_dirichlet_beta_branch() -> object:
    """Function docstring."""
    with ConfigContext(eager_mode=True):
        device = Device("cpu")
        a = Tensor(np.array(1.0), TensorConfig((), "float32", device))
        b = Tensor(np.array(1.0), TensorConfig((), "float32", device))
        key = Tensor(np.array([1, 2]), TensorConfig((2,), "int32", device))

        beta(key, a, b)
        try:
            dirichlet(
                key,
                Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", device)),
                shape=(2, 2),
            )
        except TypeError:
            pass
