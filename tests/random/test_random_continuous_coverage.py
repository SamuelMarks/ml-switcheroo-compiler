"""Tests for test_random_continuous_coverage."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core.dtype import DType


def test_random_continuous_and_transformations() -> None:
    """Verify random continuous distributions and transformations dispatch correctly."""
    distributions = [
        ("ball", (1, 2)),
        ("cauchy", (3,)),
        ("chisquare", (4,)),
        ("double_sided_maxwell", (5,)),
        ("exponential", (6,)),
        ("f", (7, 8)),
        ("generalized_normal", (9,)),
        ("gumbel", (10,)),
        ("laplace", (11,)),
        ("loggamma", (12,)),
        ("logistic", (13,)),
        ("lognormal", (14,)),
        ("maxwell", (15,)),
        ("orthogonal", (16,)),
        ("pareto", (17,)),
        ("random_gamma_p", (18,)),
        ("rayleigh", (19,)),
        ("t", (20,)),
        ("triangular", (21,)),
        ("wald", (22,)),
        ("weibull_min", (23,)),
    ]

    for name, args in distributions:
        mod = importlib.import_module(f"ml_switcheroo_compiler.random.continuous.{name}")
        with patch.object(mod, "_dispatch_random", return_value=f"mock_{name}") as mock_disp:
            func = getattr(mod, name)
            res = func(*args)
            assert res == f"mock_{name}"
            mock_disp.assert_called_once_with(name, *args)

    # Test normal, uniform, truncated_normal, gamma with _emit_random_node
    normal_mod = importlib.import_module("ml_switcheroo_compiler.random.continuous.normal")
    with patch.object(normal_mod, "_emit_random_node", return_value="mock_normal") as mock_emit:
        assert normal_mod.normal("key", (2, 2)) == "mock_normal"
        mock_emit.assert_called_with("RandomNormal", ["key"], (2, 2), DType.Float32)

        assert normal_mod.normal("key", (2, 2), dtype=DType.Float64) == "mock_normal"
        mock_emit.assert_called_with("RandomNormal", ["key"], (2, 2), DType.Float64)

    uniform_mod = importlib.import_module("ml_switcheroo_compiler.random.continuous.uniform")
    with patch.object(uniform_mod, "_emit_random_node", return_value="mock_uniform") as mock_emit:
        assert uniform_mod.uniform("key", (3, 3)) == "mock_uniform"
        mock_emit.assert_called_with("RandomUniform", ["key"], (3, 3), DType.Float32, {"minval": 0.0, "maxval": 1.0})

        assert uniform_mod.uniform("key", (3, 3), dtype=DType.Float64, minval=-1.0, maxval=2.0) == "mock_uniform"
        mock_emit.assert_called_with("RandomUniform", ["key"], (3, 3), DType.Float64, {"minval": -1.0, "maxval": 2.0})

    trunc_norm_mod = importlib.import_module("ml_switcheroo_compiler.random.continuous.truncated_normal")
    with patch.object(trunc_norm_mod, "_emit_random_node", return_value="mock_trunc") as mock_emit:
        assert trunc_norm_mod.truncated_normal("key", -2.0, 2.0, (2, 2)) == "mock_trunc"
        assert trunc_norm_mod.truncated_normal("key", -2.0, 2.0, (2, 2), dtype=DType.Float64) == "mock_trunc"

    gamma_mod = importlib.import_module("ml_switcheroo_compiler.random.continuous.gamma")
    with patch.object(gamma_mod, "_emit_random_node", return_value="mock_gamma") as mock_emit:
        assert gamma_mod.gamma("key", 1.0, (2, 2)) == "mock_gamma"
        assert gamma_mod.gamma("key", 1.0, (2, 2), dtype=DType.Float64) == "mock_gamma"

    beta_mod = importlib.import_module("ml_switcheroo_compiler.random.continuous.beta")
    with patch.object(beta_mod, "_emit_random_node", return_value="mock_beta") as mock_emit:
        assert beta_mod.beta("key", 1.0, 2.0) == "mock_beta"
        mock_emit.assert_called_with("Beta", ["key", 1.0, 2.0], (), DType.Float32)

        assert beta_mod.beta("key", 1.0, 2.0, shape=(2, 3), dtype=DType.Float64) == "mock_beta"
        mock_emit.assert_called_with("Beta", ["key", 1.0, 2.0], (2, 3), DType.Float64)

    dirichlet_mod = importlib.import_module("ml_switcheroo_compiler.random.continuous.dirichlet")
    with patch.object(dirichlet_mod, "_emit_random_node", return_value="mock_dirichlet") as mock_emit:
        assert dirichlet_mod.dirichlet("key", 1.0) == "mock_dirichlet"
        mock_emit.assert_called_with("Dirichlet", ["key", 1.0], (), DType.Float32)

        assert dirichlet_mod.dirichlet("key", 1.0, shape=(3,), dtype=DType.Float64) == "mock_dirichlet"
        mock_emit.assert_called_with("Dirichlet", ["key", 1.0], (3,), DType.Float64)

    mvn_mod = importlib.import_module("ml_switcheroo_compiler.random.continuous.multivariate_normal")
    with patch.object(mvn_mod, "_emit_random_node", return_value="mock_mvn") as mock_emit:
        assert mvn_mod.multivariate_normal("key", 0.0, 1.0) == "mock_mvn"
        mock_emit.assert_called_with("MultivariateNormal", ["key"], (), DType.Float32, {"method": "cholesky"})

        from ml_switcheroo_compiler.core.tensor import Tensor

        mean_t = MagicMock(spec=Tensor)
        cov_t = MagicMock(spec=Tensor)
        opts = mvn_mod.MultivariateNormalOptions(shape=(2,), dtype=DType.Float64, method="svd")
        assert mvn_mod.multivariate_normal("key", mean_t, cov_t, options=opts) == "mock_mvn"
        mock_emit.assert_called_with("MultivariateNormal", ["key", mean_t, cov_t], (2,), DType.Float64, {"method": "svd"})

    trans_mod = importlib.import_module("ml_switcheroo_compiler.random.transformations")
    with patch.object(trans_mod, "_emit_random_node", return_value="mock_shuffle") as mock_emit:
        mock_tensor = MagicMock()
        mock_tensor.shape = (4, 5)
        mock_tensor.dtype = "float32"
        res = trans_mod.shuffle("prng_key", mock_tensor, axis=1)
        assert res == "mock_shuffle"
        mock_emit.assert_called_once_with("RandomShuffle", ["prng_key", mock_tensor], (4, 5), "float32", {"axis": 1})
