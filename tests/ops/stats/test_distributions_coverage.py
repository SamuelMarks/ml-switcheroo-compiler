"""Tests for test_distributions_coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

import ml_switcheroo_compiler.ops.stats.distributions as dist_mod
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef


class DummyWithShape:
    """Mock operand providing shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 123


def test_statistical_distributions_full_coverage() -> None:
    """Test full coverage for statistical distribution operations and helper functions."""
    t_in = Tensor(np.ones((3, 3), dtype=np.float32), TensorConfig((3, 3), "float32", "cpu"))

    dist_pairs: list[tuple[type[OpDef], str]] = [
        (dist_mod.NormPdf, "norm_pdf"),
        (dist_mod.NormCdf, "norm_cdf"),
        (dist_mod.GammaPdf, "gamma_pdf"),
        (dist_mod.GammaCdf, "gamma_cdf"),
        (dist_mod.BetaPdf, "beta_pdf"),
        (dist_mod.BetaCdf, "beta_cdf"),
        (dist_mod.PoissonPmf, "poisson_pmf"),
        (dist_mod.PoissonCdf, "poisson_cdf"),
        (dist_mod.BinomPmf, "binom_pmf"),
        (dist_mod.BinomCdf, "binom_cdf"),
    ]

    for op_cls, fn_name in dist_pairs:
        instance = op_cls()
        assert instance.infer_shape(t_in) == (3, 3)

    with patch("ml_switcheroo_compiler.ops.stats.distributions.get_op") as mock_get_op:
        mock_inst = MagicMock()
        mock_inst.return_value = t_in
        mock_get_op.return_value = MagicMock(return_value=mock_inst)

        assert dist_mod.norm_pdf(t_in) is t_in
        assert dist_mod.norm_cdf(t_in) is t_in
        assert dist_mod.gamma_pdf(t_in, 2.0) is t_in
        assert dist_mod.gamma_cdf(t_in, 2.0) is t_in
        assert dist_mod.beta_pdf(t_in, 2.0, 3.0) is t_in
        assert dist_mod.beta_cdf(t_in, 2.0, 3.0) is t_in
        assert dist_mod.poisson_pmf(t_in, 1.5) is t_in
        assert dist_mod.poisson_cdf(t_in, 1.5) is t_in
        assert dist_mod.binom_pmf(t_in, 10, 0.5) is t_in
        assert dist_mod.binom_cdf(t_in, 10, 0.5) is t_in
