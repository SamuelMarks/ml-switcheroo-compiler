def test_random_stateless():
    from unittest.mock import MagicMock, patch

    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.random_stateless import (
        Algorithm,
        Generator,
        RandomGenerationConfig,
        create_rng_state,
        get_global_generator,
        index_shuffle,
        set_global_generator,
        stateless_beta,
        stateless_categorical,
        stateless_fold_in,
        stateless_gamma,
        stateless_parameterized_truncated_normal,
        stateless_poisson,
        stateless_random_binomial,
        stateless_random_normal,
        stateless_random_uniform,
        stateless_shuffle,
        stateless_split,
        stateless_truncated_normal,
    )

    t = Tensor(np.array([1, 2]), TensorConfig((2,), "int32", "cpu"))

    with (
        patch("ml_switcheroo_compiler.random.continuous.uniform") as mock_u,
        patch("ml_switcheroo_compiler.ops.binary.add") as mock_add,
        patch("ml_switcheroo_compiler.ops.binary.multiply") as mock_mul,
        patch("ml_switcheroo_compiler.ops.creation.full") as mock_full,
        patch("ml_switcheroo_compiler.random.continuous.normal") as mock_n,
        patch("ml_switcheroo_compiler.random.distributions_discrete.binomial") as mock_b,
        patch("ml_switcheroo_compiler.random.continuous.truncated_normal") as mock_tn,
        patch("ml_switcheroo_compiler.ops.cast") as mock_cast,
        patch("ml_switcheroo_compiler.random.distributions_discrete.categorical") as mock_cat,
        patch("ml_switcheroo_compiler.random.continuous.gamma") as mock_g,
        patch("ml_switcheroo_compiler.random.continuous.beta") as mock_beta,
        patch("ml_switcheroo_compiler.random.transformations.shuffle") as mock_shuf,
        patch("ml_switcheroo_compiler.random.distributions_discrete.poisson") as mock_poi,
        patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_gab,
        patch("ml_switcheroo_compiler.ops.shape.utils._emit_shape_node") as mock_emit,
    ):
        mock_u.return_value = t
        mock_add.return_value = t
        mock_mul.return_value = t
        mock_full.return_value = t
        mock_n.return_value = t
        mock_b.return_value = t
        mock_tn.return_value = t
        mock_cast.return_value = t
        mock_cat.return_value = t
        mock_g.return_value = t
        mock_beta.return_value = t
        mock_shuf.return_value = t
        mock_poi.return_value = t
        mock_backend = MagicMock()
        mock_backend.execute_op.return_value = "mock"
        mock_gab.return_value = mock_backend
        mock_emit.return_value = "mock_emit"

        # stateless_random_uniform
        res = stateless_random_uniform((2,), t, minval=0.0, maxval=1.0)
        assert res is t

        # stateless_random_normal
        res = stateless_random_normal((2,), t, mean=0.0, stddev=1.0)
        assert res is t

        # stateless_random_binomial
        res = stateless_random_binomial((2,), t, counts=t, probabilities=t)
        assert res is t

        # stateless_truncated_normal
        res = stateless_truncated_normal((2,), t)
        assert res is t

        # stateless_categorical
        res = stateless_categorical(t, t, 1)
        assert res is t

        # stateless_gamma
        res = stateless_gamma((2,), t, t)
        assert res is t

        # stateless_beta
        res = stateless_beta((2,), t, t, t)
        assert res is t

        # stateless_shuffle
        res = stateless_shuffle(t, t)
        assert res is t

        # stateless_parameterized_truncated_normal
        res = stateless_parameterized_truncated_normal((2,), t, RandomGenerationConfig())
        assert res is not None

        # stateless_poisson
        res = stateless_poisson((2,), t, t)
        assert res is t

        config.eager_mode = True
        index_shuffle(t, t, t)
        stateless_fold_in(t, t)
        stateless_split(t, 2)

        config.eager_mode = False
        index_shuffle(t, t, t)
        stateless_fold_in(t, t)
        stateless_split(t, 2)

        gen = Generator(state=t, alg=Algorithm.PHILOX)
        gen2 = Generator.from_seed(t, Algorithm.THREEFRY)

        config.eager_mode = True
        res = gen.normal((2,))
        assert res == "mock"

        res = gen.uniform((2,))
        assert res == "mock"

        config.eager_mode = False
        res = gen.normal((2,))
        assert res == "mock_emit"
        res = gen.uniform((2,))
        assert res == "mock_emit"

    # Others
    create_rng_state(t)
    set_global_generator(gen)
    assert get_global_generator() is gen


def test_get_global_generator_init():
    from ml_switcheroo_compiler.ops.random_stateless import _GLOBAL_GENERATOR_STATE, get_global_generator

    _GLOBAL_GENERATOR_STATE["generator"] = None
    assert get_global_generator() is not None
