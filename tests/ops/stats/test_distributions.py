"""Behavioral unit tests for distributions."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np
from pytest_mock import MockerFixture

import ml_switcheroo_compiler.ops.stats.distributions as dist_mod
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.stats.distributions import (
    BetaCdf,
    BetaPdf,
    BinomCdf,
    BinomPmf,
    GammaCdf,
    GammaPdf,
    NormCdf,
    NormPdf,
    PoissonCdf,
    PoissonPmf,
    beta_cdf,
    beta_pdf,
    binom_cdf,
    binom_pmf,
    gamma_cdf,
    gamma_pdf,
    norm_cdf,
    norm_pdf,
    poisson_cdf,
    poisson_pmf,
)
from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockTensor(MockTestStub):
    """Test class MockTensor."""

    def __init__(self, shape: BackendArg = ()) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_distributions_infer_shape() -> None:
    """Test function test_distributions_infer_shape."""
    t = MockTensor((2, 3))
    assert NormPdf().infer_shape(t) == (2, 3)
    assert NormCdf().infer_shape(t) == (2, 3)
    assert GammaPdf().infer_shape(t) == (2, 3)
    assert GammaCdf().infer_shape(t) == (2, 3)
    assert BetaPdf().infer_shape(t) == (2, 3)
    assert BetaCdf().infer_shape(t) == (2, 3)
    assert PoissonPmf().infer_shape(t) == (2, 3)
    assert PoissonCdf().infer_shape(t) == (2, 3)
    assert BinomPmf().infer_shape(t) == (2, 3)
    assert BinomCdf().infer_shape(t) == (2, 3)


def test_distributions_funcs(mocker: MockerFixture) -> None:
    """Test function test_distributions_funcs.

    Args:
        mocker: Test parameter `mocker`.
    """
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.stats.distributions.get_op")
    mock_op = mocker.MagicMock()
    mock_get_op.return_value = mocker.MagicMock(return_value=mock_op)
    assert norm_pdf(t) == mock_op()
    assert norm_cdf(t) == mock_op()
    assert gamma_pdf(t, t) == mock_op()
    assert gamma_cdf(t, t) == mock_op()
    assert beta_pdf(t, t, t) == mock_op()
    assert beta_cdf(t, t, t) == mock_op()
    assert poisson_pmf(t, t) == mock_op()
    assert poisson_cdf(t, t) == mock_op()
    assert binom_pmf(t, t, t) == mock_op()
    assert binom_cdf(t, t, t) == mock_op()


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

    for op_cls, _fn_name in dist_pairs:
        instance = op_cls()
        assert instance.dispatch_infer_shape(t_in) == (3, 3)

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
