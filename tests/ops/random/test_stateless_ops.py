"""Tests for stateless random distributions and PRNG utilities."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.ops.random as ops_random
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_stateless import (
    Algorithm,
    NormalConfig,
    RandomGenerationConfig,
    UniformConfig,
    _extract_seed_pair,
    index_shuffle,
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
from tests.testing_utils import as_tensor, as_tensor_obj

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_ops_random_unified_package() -> None:
    """Verify `ml_switcheroo_compiler.ops.random` exports all unified stateless PRNG functions."""
    assert hasattr(ops_random, "stateless_random_uniform")
    assert hasattr(ops_random, "stateless_split")
    assert hasattr(ops_random, "sobol_sample")
    assert hasattr(ops_random, "rademacher")
    assert hasattr(ops_random, "uniform")
    assert hasattr(ops_random, "normal")


def test_stateless_random_uniform(mocker: EagerTensorProtocol) -> None:
    """Verify `stateless_random_uniform` delegates to uniform, multiply, and add.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.continuous.uniform", return_value="unif")
    mocker.patch("ml_switcheroo_compiler.ops.creation.full", return_value="full")
    mocker.patch("ml_switcheroo_compiler.ops.binary.multiply", return_value="mult")
    mocker.patch("ml_switcheroo_compiler.ops.binary.add", return_value="add")

    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    with ConfigContext(eager_mode=True):
        res = stateless_random_uniform((2, 3), seed, 1.0, 5.0, "float32")
    assert res == "add"

    res2 = stateless_random_uniform((2, 3), seed, 1.0, 5.0, DType("float32"))
    assert res2 == "add"


def test_stateless_random_normal(mocker: EagerTensorProtocol) -> None:
    """Verify `stateless_random_normal` delegates to normal, multiply, and add.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.continuous.normal", return_value="norm")
    mocker.patch("ml_switcheroo_compiler.ops.creation.full", return_value="full")
    mocker.patch("ml_switcheroo_compiler.ops.binary.multiply", return_value="mult")
    mocker.patch("ml_switcheroo_compiler.ops.binary.add", return_value="add")

    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    res = stateless_random_normal((2, 3), seed, 1.0, 5.0, "float32")
    assert res == "add"

    res2 = stateless_random_normal((2, 3), seed, 1.0, 5.0, DType("float32"))
    assert res2 == "add"


def test_stateless_random_binomial(mocker: EagerTensorProtocol) -> None:
    """Verify `stateless_random_binomial` delegates to discrete binomial.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete.binomial", return_value="binom")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    res = stateless_random_binomial((2, 3), seed, 10.0, 0.5, "int32")
    assert res == "binom"
    res2 = stateless_random_binomial((2, 3), seed, 10.0, 0.5, DType("int32"))
    assert res2 == "binom"


def test_stateless_truncated_normal(mocker: EagerTensorProtocol) -> None:
    """Verify `stateless_truncated_normal` delegates to truncated_normal, multiply, and add.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.continuous.truncated_normal", return_value="trunc")
    mocker.patch("ml_switcheroo_compiler.ops.creation.full", return_value="full")
    mocker.patch("ml_switcheroo_compiler.ops.binary.multiply", return_value="mult")
    mocker.patch("ml_switcheroo_compiler.ops.binary.add", return_value="add")

    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    res = stateless_truncated_normal((2, 3), seed, 1.0, 5.0, "float32")
    assert res == "add"
    res2 = stateless_truncated_normal((2, 3), seed, 1.0, 5.0, DType("float32"))
    assert res2 == "add"


def test_stateless_categorical(mocker: EagerTensorProtocol) -> None:
    """Verify `stateless_categorical` delegates to categorical and cast.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete.categorical", return_value="cat")
    mocker.patch("ml_switcheroo_compiler.ops.cast", return_value="typing.cast")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    logits = Tensor([[1.0, 2.0]], TensorConfig((1, 2), "float32", "cpu"))
    res = stateless_categorical(logits, 5, seed, "int32")
    assert res == "typing.cast"
    res2 = stateless_categorical(logits, 5, seed, DType("int32"))
    assert res2 == "typing.cast"

    logits_no_shape = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    res3 = stateless_categorical(logits_no_shape, 5, seed, "int32")
    assert res3 == "typing.cast"


def test_stateless_gamma(mocker: EagerTensorProtocol) -> None:
    """Verify `stateless_gamma` delegates to continuous gamma.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.continuous.gamma", return_value="gamma")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    alpha = Tensor(2.0, TensorConfig((), "float32", "cpu"))
    res = stateless_gamma((2, 3), seed, alpha, "float32")
    assert res == "gamma"
    res2 = stateless_gamma((2, 3), seed, alpha, DType("float32"))
    assert res2 == "gamma"


def test_stateless_beta(mocker: EagerTensorProtocol) -> None:
    """Verify `stateless_beta` delegates to continuous beta.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.continuous.beta", return_value="beta")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    alpha = Tensor(2.0, TensorConfig((), "float32", "cpu"))
    beta_param = Tensor(3.0, TensorConfig((), "float32", "cpu"))
    res = stateless_beta((2, 3), seed, alpha, beta_param, "float32")
    assert res == "beta"
    res2 = stateless_beta((2, 3), seed, alpha, beta_param, DType("float32"))
    assert res2 == "beta"


def test_stateless_shuffle(mocker: EagerTensorProtocol) -> None:
    """Verify `stateless_shuffle` delegates to transformations.shuffle.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.transformations.shuffle", return_value="shuffle")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    x = Tensor([1, 2, 3], TensorConfig((3,), "int32", "cpu"))
    res = stateless_shuffle(x, seed, 0)
    assert res == "shuffle"


def test_stateless_parameterized_truncated_normal() -> None:
    """Verify `stateless_parameterized_truncated_normal` produces bounded samples."""
    orig = config.eager_mode
    config.eager_mode = True
    try:
        seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
        cfg = RandomGenerationConfig(means=1.0, stddevs=0.5, minvals=-1.0, maxvals=1.0)
        res = as_tensor_obj(stateless_parameterized_truncated_normal((2,), seed, cfg))
        assert res.config.shape == (2,)
        arr = np.asarray(res.data)
        assert np.all(arr >= 0.5)
        assert np.all(arr <= 1.5)
        res_empty = as_tensor_obj(stateless_parameterized_truncated_normal((), seed, cfg))
        assert res_empty.config.shape == ()
    finally:
        config.eager_mode = orig


def test_algorithms_and_configs() -> None:
    """Verify Algorithm constants and distribution config dataclasses."""
    assert Algorithm.PHILOX == 1
    assert Algorithm.THREEFRY == 2
    assert Algorithm.AUTO_SELECT == 3
    nc = NormalConfig(mean=2.0, stddev=3.0)
    assert nc.mean == 2.0 and nc.stddev == 3.0
    uc = UniformConfig(minval=-1.0, maxval=2.0)
    assert uc.minval == -1.0 and uc.maxval == 2.0


def test_extract_seed_pair_edge_cases() -> None:
    """Verify `_extract_seed_pair` across int, str, nested list, and empty inputs."""
    p_int = _extract_seed_pair(42)
    assert p_int[0] == 42
    p_str = _extract_seed_pair("abc")
    assert isinstance(p_str[0], int)
    p_list = _extract_seed_pair(["a", [1, "b"]])
    assert isinstance(p_list[0], int)
    p_empty = _extract_seed_pair([])
    assert p_empty == (0, 0x9E3779B9)


def test_index_shuffle_and_fold_in() -> None:
    """Verify `index_shuffle` and `stateless_fold_in` across scalar, array, and fallback inputs."""
    orig = config.eager_mode
    config.eager_mode = True
    try:
        res = index_shuffle(1, "seed", 10)
        assert isinstance(res, int) and 0 <= res <= 10

        scalar_arr = Tensor(np.array(2, dtype=np.int32), TensorConfig((), "int32", "cpu"))
        res_scalar = as_tensor_obj(index_shuffle(scalar_arr, 123, np.array(5)))
        assert res_scalar.shape == ()

        vec_arr = Tensor(np.array([1, 2], dtype=np.int32), TensorConfig((2,), "int32", "cpu"))
        res_vec = as_tensor_obj(index_shuffle(vec_arr, 123, [5]))
        assert res_vec.shape == (2,)
        res_list = as_tensor_obj(index_shuffle([1, 2], 123, 5))
        assert res_list.shape == ()
        res_empty_max = index_shuffle(vec_arr, 123, [])
        assert res_empty_max is vec_arr

        res_non_tensor = index_shuffle("raw_idx", 123, 5)
        assert res_non_tensor == "raw_idx"

        res_fold = stateless_fold_in("seed", "data")
        assert isinstance(res_fold, str) and res_fold.startswith("seed_")
    finally:
        config.eager_mode = orig


def test_stateless_split() -> None:
    """Verify `stateless_split` in eager mode."""
    orig = config.eager_mode
    config.eager_mode = True
    try:
        res = stateless_split("seed", 3)
        assert as_tensor(getattr(res, "shape")) == (3, 2)
    finally:
        config.eager_mode = orig


def test_stateless_poisson(mocker: EagerTensorProtocol) -> None:
    """Verify `stateless_poisson` delegates to discrete poisson.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.distributions_discrete.poisson", return_value="poisson")
    seed = Tensor([0, 1], TensorConfig((2,), "int64", "cpu"))
    lam = Tensor(2.0, TensorConfig((), "float32", "cpu"))
    res = stateless_poisson((2, 3), seed, lam, "int32")
    assert res == "poisson"
    res2 = stateless_poisson((2, 3), seed, lam, DType("int32"))
    assert res2 == "poisson"
