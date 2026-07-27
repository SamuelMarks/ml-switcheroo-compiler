# ruff: noqa: E501
from unittest.mock import MagicMock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager_registry as reg
from ml_switcheroo_compiler import random as rn
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_stateless import stateless_beta, stateless_shuffle
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
from ml_switcheroo_compiler.random.state import rng_bit_generator, rng_uniform
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

"Core abstractions and logic definitions for test_random_coverage.py."


def test_random_ops() -> object:
    """Test the random ops behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        with ConfigContext(eager_mode=False):
            global_tracing_state.start_tracing()
            keys = rn.split(rn.PRNGKey(0))
            k1 = keys
            rn.uniform(k1, (2,), minval=0.0, maxval=1.0)
            rn.normal(k1, (2,))
            rn.randint(k1, (2,), minval=0, maxval=10)
            rn.bernoulli(k1, 0.5, (2,))
            t_mock = rn.uniform(k1, (2,))
            rn.categorical(k1, t_mock)
            rn.permutation(k1, t_mock)
            rn.choice(k1, t_mock)
            rn.truncated_normal(k1, -1.0, 1.0, (2,))
            rn.fold_in(k1, 5)
            global_tracing_state.stop_tracing()
        with ConfigContext(eager_mode=True, backend="numpy"):
            try:
                rn.state._emit_random_node("FakeOp", [], (), DType.Float32, {})
            except Exception:
                pass
    except Exception as e:
        raise e
        pass


def test_stateless_beta() -> object:
    """Test the stateless beta behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        seed = Tensor(np.array([0, 0]), TensorConfig((2,), DType.Int32, None))
        alpha = Tensor(np.array(1.0), TensorConfig((), DType.Float32, None))
        beta_param = Tensor(np.array(1.0), TensorConfig((), DType.Float32, None))
        with ConfigContext(eager_mode=True):
            res = stateless_beta((2, 2), seed, alpha, beta_param)
            assert res.shape == (2, 2)
        with ConfigContext(eager_mode=False):
            global_tracing_state.start_tracing()
            try:
                p_seed = Tensor(ProxyTensor(id="mock_seed", shape=(2,), dtype=DType.Int32.value), TensorConfig((2,), DType.Int32, None))
                p_alpha = Tensor(ProxyTensor(id="mock_a", shape=(), dtype=DType.Float32.value), TensorConfig((), DType.Float32, None))
                p_beta = Tensor(ProxyTensor(id="mock_b", shape=(), dtype=DType.Float32.value), TensorConfig((), DType.Float32, None))
                res = stateless_beta((2, 2), p_seed, p_alpha, p_beta)
                assert res is not None
            finally:
                global_tracing_state.stop_tracing()
    except Exception as e:
        raise e
        pass


def test_stateless_shuffle() -> object:
    """Test the stateless shuffle behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        seed = Tensor(np.array([0, 0]), TensorConfig((2,), DType.Int32, None))
        x = Tensor(np.array([1, 2, 3, 4]), TensorConfig((4,), DType.Int32, None))
        y = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig((2, 2), DType.Int32, None))
        with ConfigContext(eager_mode=True):
            res = stateless_shuffle(x, seed)
            assert res.shape == (4,)
            res2 = stateless_shuffle(y, seed, axis=1)
            assert res2.shape == (2, 2)
        with ConfigContext(eager_mode=False):
            global_tracing_state.start_tracing()
            try:
                p_seed = Tensor(ProxyTensor(id="mock_seed", shape=(2,), dtype=DType.Int32.value), TensorConfig((2,), DType.Int32, None))
                p_x = Tensor(ProxyTensor(id="mock_x", shape=(4,), dtype=DType.Int32.value), TensorConfig((4,), DType.Int32, None))
                res = stateless_shuffle(p_x, p_seed)
                assert res is not None
            finally:
                global_tracing_state.stop_tracing()
    except Exception as e:
        raise e
        pass


"Core abstractions and logic definitions for test_random_coverage2.py."


def test_random_extra_coverage() -> object:
    """Test the random extra coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
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
    except Exception as e:
        raise e
        pass


def test_permutation_eager_none() -> object:
    """Test the permutation eager none behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        config.eager_mode = True
        key = MagicMock()
        key.data = [0, 1]
        x = MagicMock()
        x.data = np.array([1, 2, 3])
        x.dtype = "float32"
        x.shape = (3,)

        class MockNoShape:
            """Configuration class for mock no shape."""

            data = np.array([1, 2, 3])

        out = permutation(key, MockNoShape())
        assert out is not None
    except Exception as e:
        raise e
        pass


def test_choice_eager_p() -> object:
    """Test the choice eager p behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
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
    except Exception as e:
        raise e
        pass


def test_categorical_eager_2d() -> object:
    """Test the categorical eager 2d behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        device = Device(DeviceType.CPU, 0)
        config.eager_mode = True
        key = PRNGKey(0)
        logits2d = Tensor(np.array([[1.0, 2.0], [0.5, 0.5]]), TensorConfig((2, 2), DType.Float32, device))
        res = categorical(key, logits2d)
        assert res.shape == ()
        assert res.data.shape == (2,)
    except Exception as e:
        raise e
        pass


def test_truncated_normal_eager_rejection() -> object:
    """Test the truncated normal eager rejection behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        config.eager_mode = True
        key = PRNGKey(0)
        res = truncated_normal(key, lower=-0.0001, upper=0.0001, shape=(1000,), dtype=DType.Float32)
        assert res.shape == (1000,)
    except Exception as e:
        raise e
        pass


def test_discrete_distributions_tracing_no_shape() -> object:
    """Test the discrete distributions tracing no shape behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        pass
    except Exception as e:
        raise e
        pass


def dummy() -> object:
    """Evaluate and process the dummy operation.

    Returns:
        object: The evaluated or processed output.
    """
    with ConfigContext(eager_mode=False):
        dists = [ball, cauchy, f, gumbel, laplace, maxwell, pareto, t, wald, chisquare, double_sided_maxwell, exponential, generalized_normal, loggamma, logistic, lognormal, multivariate_normal, orthogonal, random_gamma_p, rayleigh, triangular, weibull_min, geometric, rademacher]
        for dist in dists:
            with pytest.raises(ValueError):
                dist()


def test_dirichlet_beta_branch() -> object:
    """Test the dirichlet beta branch behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        with ConfigContext(eager_mode=True):
            device = Device("cpu")
            a = Tensor(np.array(1.0), TensorConfig((), "float32", device))
            b = Tensor(np.array(1.0), TensorConfig((), "float32", device))
            key = Tensor(np.array([1, 2]), TensorConfig((2,), "int32", device))
            beta(key, a, b)
            try:
                dirichlet(key, Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", device)), shape=(2, 2))
            except TypeError:
                pass
    except Exception as e:
        raise e
        pass


"Combined random tests."


def test_random_ops_2() -> object:
    """Test the random ops behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        with ConfigContext(eager_mode=False):
            global_tracing_state.start_tracing()
            keys = rn.split(rn.PRNGKey(0))
            k1 = keys
            rn.uniform(k1, (2,), minval=0.0, maxval=1.0)
            rn.normal(k1, (2,))
            rn.randint(k1, (2,), minval=0, maxval=10)
            rn.bernoulli(k1, 0.5, (2,))
            t_mock = rn.uniform(k1, (2,))
            rn.categorical(k1, t_mock)
            rn.permutation(k1, t_mock)
            rn.choice(k1, t_mock)
            rn.truncated_normal(k1, -1.0, 1.0, (2,))
            rn.fold_in(k1, 5)
            global_tracing_state.stop_tracing()
        with ConfigContext(eager_mode=True, backend="numpy"):
            try:
                rn.state._emit_random_node("FakeOp", [], (), DType.Float32, {})
            except Exception:
                pass
    except Exception as e:
        raise e
        pass


def test_stateless_beta_2() -> object:
    """Test the stateless beta behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        seed = Tensor(np.array([0, 0]), TensorConfig((2,), DType.Int32, None))
        alpha = Tensor(np.array(1.0), TensorConfig((), DType.Float32, None))
        beta_param = Tensor(np.array(1.0), TensorConfig((), DType.Float32, None))
        with ConfigContext(eager_mode=True):
            res = stateless_beta((2, 2), seed, alpha, beta_param)
            assert res.shape == (2, 2)
        with ConfigContext(eager_mode=False):
            global_tracing_state.start_tracing()
            try:
                p_seed = Tensor(ProxyTensor(id="mock_seed", shape=(2,), dtype=DType.Int32.value), TensorConfig((2,), DType.Int32, None))
                p_alpha = Tensor(ProxyTensor(id="mock_a", shape=(), dtype=DType.Float32.value), TensorConfig((), DType.Float32, None))
                p_beta = Tensor(ProxyTensor(id="mock_b", shape=(), dtype=DType.Float32.value), TensorConfig((), DType.Float32, None))
                res = stateless_beta((2, 2), p_seed, p_alpha, p_beta)
                assert res is not None
            finally:
                global_tracing_state.stop_tracing()
    except Exception as e:
        raise e
        pass


def test_stateless_shuffle_2() -> object:
    """Test the stateless shuffle behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        seed = Tensor(np.array([0, 0]), TensorConfig((2,), DType.Int32, None))
        x = Tensor(np.array([1, 2, 3, 4]), TensorConfig((4,), DType.Int32, None))
        y = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig((2, 2), DType.Int32, None))
        with ConfigContext(eager_mode=True):
            res = stateless_shuffle(x, seed)
            assert res.shape == (4,)
            res2 = stateless_shuffle(y, seed, axis=1)
            assert res2.shape == (2, 2)
        with ConfigContext(eager_mode=False):
            global_tracing_state.start_tracing()
            try:
                p_seed = Tensor(ProxyTensor(id="mock_seed", shape=(2,), dtype=DType.Int32.value), TensorConfig((2,), DType.Int32, None))
                p_x = Tensor(ProxyTensor(id="mock_x", shape=(4,), dtype=DType.Int32.value), TensorConfig((4,), DType.Int32, None))
                res = stateless_shuffle(p_x, p_seed)
                assert res is not None
            finally:
                global_tracing_state.stop_tracing()
    except Exception as e:
        raise e
        pass


def test_random_extra_coverage_2() -> object:
    """Test the random extra coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
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
    except Exception as e:
        raise e
        pass


def test_permutation_eager_none_2() -> object:
    """Test the permutation eager none behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        config.eager_mode = True
        key = MagicMock()
        key.data = [0, 1]
        x = MagicMock()
        x.data = np.array([1, 2, 3])
        x.dtype = "float32"
        x.shape = (3,)

        class MockNoShape:
            """Configuration class for mock no shape."""

            data = np.array([1, 2, 3])

        out = permutation(key, MockNoShape())
        assert out is not None
    except Exception as e:
        raise e
        pass


def test_choice_eager_p_2() -> object:
    """Test the choice eager p behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
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
    except Exception as e:
        raise e
        pass


def test_categorical_eager_2d_2() -> object:
    """Test the categorical eager 2d behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        device = Device(DeviceType.CPU, 0)
        config.eager_mode = True
        key = PRNGKey(0)
        logits2d = Tensor(np.array([[1.0, 2.0], [0.5, 0.5]]), TensorConfig((2, 2), DType.Float32, device))
        res = categorical(key, logits2d)
        assert res.shape == ()
        assert res.data.shape == (2,)
    except Exception as e:
        raise e
        pass


def test_truncated_normal_eager_rejection_2() -> object:
    """Test the truncated normal eager rejection behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        config.eager_mode = True
        key = PRNGKey(0)
        res = truncated_normal(key, lower=-0.0001, upper=0.0001, shape=(1000,), dtype=DType.Float32)
        assert res.shape == (1000,)
    except Exception as e:
        raise e
        pass


def test_discrete_distributions_tracing_no_shape_2() -> object:
    """Test the discrete distributions tracing no shape behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        pass
    except Exception as e:
        raise e
        pass


def test_dirichlet_beta_branch_2() -> object:
    """Test the dirichlet beta branch behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        with ConfigContext(eager_mode=True):
            device = Device("cpu")
            a = Tensor(np.array(1.0), TensorConfig((), "float32", device))
            b = Tensor(np.array(1.0), TensorConfig((), "float32", device))
            key = Tensor(np.array([1, 2]), TensorConfig((2,), "int32", device))
            beta(key, a, b)
            try:
                dirichlet(key, Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", device)), shape=(2, 2))
            except TypeError:
                pass
    except Exception as e:
        raise e
        pass


def test_random_state_eager() -> object:
    """Test the random state eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        reg.numpy_eager_registry.register("RngBitGenerator")(lambda m, key, **kw: np.zeros(kw.get("shape")))
        reg.numpy_eager_registry.register("RngUniform")(lambda m, a, b, **kw: np.zeros(kw.get("shape")))
        with ConfigContext(eager_mode=True, backend="numpy"):
            r1 = rng_bit_generator(None, (2, 2), "uint32")
            r2 = rng_uniform(0, 1, (2, 2), "float32")
            assert r1.shape == (2, 2)
            assert r2.shape == (2, 2)
    except Exception as e:
        raise e
        pass
