"""Tests for test_random_stateless."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_random_stateless_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_random_stateless.

    Args:
        _state: Shared test state dictionary.
    """
    MagicMock, mock_add, mock_b, mock_beta = _state["MagicMock"], _state["mock_add"], _state["mock_b"], _state["mock_beta"]
    mock_cast, mock_cat, mock_emit, mock_full = _state["mock_cast"], _state["mock_cat"], _state["mock_emit"], _state["mock_full"]
    mock_g, mock_gab, mock_mul, mock_n = _state["mock_g"], _state["mock_gab"], _state["mock_mul"], _state["mock_n"]
    mock_poi, mock_shuf, mock_tn, mock_u = _state["mock_poi"], _state["mock_shuf"], _state["mock_tn"], _state["mock_u"]
    stateless_random_uniform, t = _state["stateless_random_uniform"], _state["t"]
    setattr(mock_u, "return_value", t)
    setattr(mock_add, "return_value", t)
    setattr(mock_mul, "return_value", t)
    setattr(mock_full, "return_value", t)
    setattr(mock_n, "return_value", t)
    setattr(mock_b, "return_value", t)
    setattr(mock_tn, "return_value", t)
    setattr(mock_cast, "return_value", t)
    setattr(mock_cat, "return_value", t)
    setattr(mock_g, "return_value", t)
    setattr(mock_beta, "return_value", t)
    setattr(mock_shuf, "return_value", t)
    setattr(mock_poi, "return_value", t)
    mock_backend = MagicMock()
    setattr(mock_backend.execute_op, "return_value", "mock")
    setattr(mock_gab, "return_value", mock_backend)
    setattr(mock_emit, "return_value", "mock_emit")

    # stateless_random_uniform
    res = stateless_random_uniform((2,), t, minval=0.0, maxval=1.0)
    getattr(_state, "update")({"mock_backend": mock_backend, "res": res})
    _state.update(locals())


def _test_random_stateless_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_random_stateless.

    Args:
        _state: Shared test state dictionary.
    """
    RandomGenerationConfig, stateless_beta, stateless_categorical, stateless_gamma = _state["RandomGenerationConfig"], _state["stateless_beta"], _state["stateless_categorical"], _state["stateless_gamma"]
    stateless_parameterized_truncated_normal, stateless_poisson, stateless_random_binomial, stateless_random_normal = _state["stateless_parameterized_truncated_normal"], _state["stateless_poisson"], _state["stateless_random_binomial"], _state["stateless_random_normal"]
    stateless_shuffle, stateless_truncated_normal, t = _state["stateless_shuffle"], _state["stateless_truncated_normal"], _state["t"]
    pass

    # stateless_random_normal
    res = stateless_random_normal((2,), t, mean=0.0, stddev=1.0)
    pass

    # stateless_random_binomial
    res = stateless_random_binomial((2,), t, counts=t, probabilities=t)
    pass

    # stateless_truncated_normal
    res = stateless_truncated_normal((2,), t)
    pass

    # stateless_categorical
    res = stateless_categorical(t, t, 1)
    pass

    # stateless_gamma
    res = stateless_gamma((2,), t, t)
    pass

    # stateless_beta
    res = stateless_beta((2,), t, t, t)
    pass

    # stateless_shuffle
    res = stateless_shuffle(t, t)
    pass

    # stateless_parameterized_truncated_normal
    res = stateless_parameterized_truncated_normal((2,), t, RandomGenerationConfig())
    assert res is not None

    # stateless_poisson
    res = stateless_poisson((2,), t, t)
    getattr(_state, "update")({"res": res})
    _state.update(locals())


def _test_random_stateless_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_random_stateless.

    Args:
        _state: Shared test state dictionary.
    """
    Algorithm, Generator, config, index_shuffle = _state["Algorithm"], _state["Generator"], _state["config"], _state["index_shuffle"]
    stateless_fold_in, stateless_split, t = _state["stateless_fold_in"], _state["stateless_split"], _state["t"]
    pass

    setattr(config, "eager_mode", True)
    index_shuffle(t, t, t)
    stateless_fold_in(t, t)
    stateless_split(t, 2)

    setattr(config, "eager_mode", False)
    index_shuffle(t, t, t)
    stateless_fold_in(t, t)
    stateless_split(t, 2)

    gen = Generator(state=t, alg=Algorithm.PHILOX)
    Generator.from_seed(t, Algorithm.THREEFRY)

    setattr(config, "eager_mode", True)
    res = gen.normal((2,))
    assert res == "mock"

    res = gen.uniform((2,))
    assert res == "mock"

    setattr(config, "eager_mode", False)
    res = gen.normal((2,))
    getattr(_state, "update")({"gen": gen, "res": res})
    _state.update(locals())


def _test_random_stateless_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_random_stateless.

    Args:
        _state: Shared test state dictionary.
    """
    gen, res = _state["gen"], _state["res"]
    assert res == "mock_emit"
    res = gen.uniform((2,))
    assert res == "mock_emit"
    getattr(_state, "update")({"res": res})
    _state.update(locals())


def test_random_stateless() -> None:
    """Test function test_random_stateless."""
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
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"Algorithm": Algorithm, "Generator": Generator, "MagicMock": MagicMock, "RandomGenerationConfig": RandomGenerationConfig})
        getattr(_state, "update")({"Tensor": Tensor, "TensorConfig": TensorConfig, "config": config, "create_rng_state": create_rng_state})
        getattr(_state, "update")({"get_global_generator": get_global_generator, "index_shuffle": index_shuffle, "mock_add": mock_add, "mock_b": mock_b})
        getattr(_state, "update")({"mock_beta": mock_beta, "mock_cast": mock_cast, "mock_cat": mock_cat, "mock_emit": mock_emit})
        getattr(_state, "update")({"mock_full": mock_full, "mock_g": mock_g, "mock_gab": mock_gab, "mock_mul": mock_mul})
        getattr(_state, "update")({"mock_n": mock_n, "mock_poi": mock_poi, "mock_shuf": mock_shuf, "mock_tn": mock_tn})
        getattr(_state, "update")({"mock_u": mock_u, "np": np, "patch": patch, "set_global_generator": set_global_generator})
        getattr(_state, "update")({"stateless_beta": stateless_beta, "stateless_categorical": stateless_categorical, "stateless_fold_in": stateless_fold_in, "stateless_gamma": stateless_gamma})
        getattr(_state, "update")({"stateless_parameterized_truncated_normal": stateless_parameterized_truncated_normal, "stateless_poisson": stateless_poisson, "stateless_random_binomial": stateless_random_binomial, "stateless_random_normal": stateless_random_normal})
        getattr(_state, "update")({"stateless_random_uniform": stateless_random_uniform, "stateless_shuffle": stateless_shuffle, "stateless_split": stateless_split, "stateless_truncated_normal": stateless_truncated_normal})
        getattr(_state, "update")({"t": t})
        for _sub_fn in (
            _test_random_stateless_sub_1,
            _test_random_stateless_sub_2,
            _test_random_stateless_sub_3,
            _test_random_stateless_sub_4,
        ):
            _sub_fn(_state)
        assert isinstance(_state, dict)
        gen = _state["gen"]

    # Others
    create_rng_state(t)
    set_global_generator(gen)
    assert get_global_generator() is gen


def test_get_global_generator_init() -> None:
    """Test function test_get_global_generator_init."""
    from ml_switcheroo_compiler.ops.random_stateless import _GLOBAL_GENERATOR_STATE, get_global_generator

    _GLOBAL_GENERATOR_STATE["generator"] = None
    assert get_global_generator() is not None
