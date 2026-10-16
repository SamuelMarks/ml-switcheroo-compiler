"""Tests for test_distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.stats.distributions import BetaCdf, BetaPdf, BinomCdf, BinomPmf, GammaCdf, GammaPdf, NormCdf, NormPdf, PoissonCdf, PoissonPmf, beta_cdf, beta_pdf, binom_cdf, binom_pmf, gamma_cdf, gamma_pdf, norm_cdf, norm_pdf, poisson_cdf, poisson_pmf

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockTensor:
    """Test class MockTensor."""

    def __init__(self, shape: EagerValue = ()) -> None:
        """Test function __init__."""
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_distributions_infer_shape() -> None:
    """Test function test_distributions_infer_shape."""
    t = MockTensor((2, 3))
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", NormPdf)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", NormCdf)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", GammaPdf)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", GammaCdf)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", BetaPdf)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", BetaCdf)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", PoissonPmf)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", PoissonCdf)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", BinomPmf)().infer_shape)(t) == (2, 3)
    assert cast("Callable[..., EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", BinomCdf)().infer_shape)(t) == (2, 3)


def test_distributions_funcs(mocker: EagerTensorProtocol) -> None:
    """Test function test_distributions_funcs."""
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), "float32", "cpu"))
    mock_get_op = mocker.patch("ml_switcheroo_compiler.ops.stats.distributions.get_op")
    mock_op = mocker.MagicMock()
    setattr(mock_get_op, "return_value", mocker.MagicMock(return_value=mock_op))
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
