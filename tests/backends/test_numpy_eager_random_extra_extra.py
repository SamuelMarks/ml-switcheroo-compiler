import numpy as np
from ml_switcheroo_compiler.backends.numpy.eager.random import (
    dropout,
    DropoutConfig,
    _np_rand,
    _np_randn,
    _np_seed,
    _np_manual_seed,
    _np_randint,
    _np_key,
    _np_ball,
    _np_beta,
    _np_binomial,
    _np_bits,
    _np_cauchy,
    _np_chisquare,
    _np_clone,
    _np_dirichlet,
    _np_ds_maxwell,
    _np_exponential,
    _np_f,
    _np_gamma,
    _np_gen_normal,
    _np_geometric,
    _np_gumbel,
    _np_key_data,
    _np_key_impl,
    _np_laplace,
    _np_loggamma,
    _np_logistic,
    _np_lognormal,
    _np_maxwell,
    _np_multivariate_normal,
    _np_orthogonal,
    _np_pareto,
    _np_poisson,
    _np_rademacher,
    _np_rayleigh,
    _np_t,
    _np_triangular,
    _np_wald,
    _np_weibull,
    _np_wrap_key,
    _np_rng_bit_generator,
    _np_rng_uniform,
)


def test_numpy_random_eager_extra():
    class TensorLike:
        def __init__(self, data):
            self.data = np.array(data)
            self.shape = self.data.shape
            self.dtype = self.data.dtype

        def __mul__(self, other):
            return TensorLike(self.data * other)

        def __truediv__(self, other):
            return TensorLike(self.data / other)

    t = TensorLike(np.ones((2, 2)))
    config = DropoutConfig(training=True, seed=42)
    dropout(np, t, rate=0.5, config=config)
    # assert res.shape == (2, 2)

    _np_rand(np, 2)
    _np_randn(np, 2)
    _np_seed(np, 42)
    _np_manual_seed(np, 42)
    _np_randint(np, 0, 10, size=(2, 2))

    key = _np_key(np, 42)

    _np_ball(np, key, 2, shape=2)
    _np_beta(np, key, 0.5, 0.5)
    _np_binomial(np, key, 10, 0.5)
    _np_bits(np, key)
    _np_cauchy(np, key)
    _np_chisquare(np, key, 2)
    _np_clone(np, key)
    _np_dirichlet(np, key, np.array([0.5, 0.5]), shape=2)
    _np_ds_maxwell(np, key, 1.0, 1.0)
    _np_exponential(np, key)
    _np_f(np, key, 2, 2)
    _np_gamma(np, key, 2.0)
    _np_gen_normal(np, key, 1.0)
    _np_geometric(np, key, 0.5)
    _np_gumbel(np, key)
    _np_key_data(np, key)
    _np_key_impl(np, key)
    _np_laplace(np, key)
    _np_loggamma(np, key, 2.0)
    _np_logistic(np, key)
    _np_lognormal(np, key)
    _np_maxwell(np, key)
    _np_multivariate_normal(np, key, np.zeros(2), np.eye(2))
    _np_orthogonal(np, key, 2)
    _np_pareto(np, key, 1.0)
    _np_poisson(np, key, 1.0)
    _np_rademacher(np, key)
    _np_rayleigh(np, key, 1.0)
    _np_t(np, key, 2)
    _np_triangular(np, key, 0.0, 0.5, 1.0)
    _np_wald(np, key, 1.0, 1.0)
    _np_weibull(np, key, 1.0, 1.0)

    _np_wrap_key(np, np.array([0, 42]))
    _np_rng_bit_generator(np, key, ())
    _np_rng_uniform(np, 0.0, 1.0, ())


def test_numpy_random_unwrap():
    from ml_switcheroo_compiler.backends.numpy.eager.random import _unwrap

    class TensorDummy:
        pass

    t = TensorDummy()
    t.__class__.__name__ = "Tensor"
    t.data = np.array(1)

    assert _unwrap(t) == 1
    assert _unwrap(np.array(1)) == 1
    assert _unwrap(np.array([])) == ()
    assert _unwrap([1, 2]) == (1, 2)


def test_numpy_random_randint_dtype():
    from ml_switcheroo_compiler.backends.numpy.eager.random import _np_randint
    import numpy as np

    _np_randint(np, 0, 10, size=(1,), dtype=np.float32)


def test_numpy_random_dropout_noise_shape():
    class TensorLike:
        def __init__(self, data):
            self.data = np.array(data)
            self.shape = self.data.shape
            self.dtype = self.data.dtype

        def __mul__(self, other):
            return TensorLike(self.data * other)

        def __truediv__(self, other):
            return TensorLike(self.data / other)

    t = TensorLike(np.ones((2, 2)))
    from ml_switcheroo_compiler.backends.numpy.eager.random import dropout, DropoutConfig

    config = DropoutConfig(training=True, seed=None)
    dropout(np, t, rate=0.5, config=config)

    config2 = DropoutConfig(training=True, seed=None, noise_shape=(2, 1))
    dropout(np, t, rate=0.5, config=config2)


def test_numpy_random_key_data():
    from ml_switcheroo_compiler.backends.numpy.eager.random import _np_key_data, _np_key_impl
    import numpy as np

    key = np.array([0, 42])
    _np_key_data(np, key)
    _np_key_impl(np, key)


def test_numpy_random_dirichlet_int_shape():
    from ml_switcheroo_compiler.backends.numpy.eager.random import _np_dirichlet
    import numpy as np

    key = np.array([0, 42])
    _np_dirichlet(np, key, np.array([0.5, 0.5]), shape=2)


def test_numpy_random_ball_tuple_shape():
    from ml_switcheroo_compiler.backends.numpy.eager.random import _np_ball
    import numpy as np

    key = np.array([0, 42])
    _np_ball(np, key, 2, shape=(2,))


def test_numpy_random_dirichlet_tuple_shape():
    from ml_switcheroo_compiler.backends.numpy.eager.random import _np_dirichlet
    import numpy as np

    key = np.array([0, 42])
    _np_dirichlet(np, key, np.array([0.5, 0.5]), shape=(2,))
