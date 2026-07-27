# ruff: noqa: E501
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.stats.distributions import BetaCdf, BetaPdf, BinomCdf, BinomPmf, GammaCdf, GammaPdf, NormCdf, NormPdf, PoissonCdf, PoissonPmf, beta_cdf, beta_pdf, binom_cdf, binom_pmf, gamma_cdf, gamma_pdf, norm_cdf, norm_pdf, poisson_cdf, poisson_pmf


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"
        self.data = [1, 2]


def test_distributions_infer_shape():
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


def test_distributions_funcs(mocker):
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
