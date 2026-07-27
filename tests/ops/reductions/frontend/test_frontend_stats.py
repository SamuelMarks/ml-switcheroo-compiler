import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_stats import approx_max_k, approx_min_k, corrcoef, correlate, cov, ctc_loss, pmean, psum


def test_psum(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value="psum")
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert psum(x, "i") == "psum"


def test_pmean(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value="pmean")
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert pmean(x, "i") == "pmean"


def test_approx_max_k(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", side_effect=["val", "idx"])
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert approx_max_k(x, 5) == ("val", "idx")


def test_approx_min_k(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", side_effect=["val", "idx"])
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert approx_min_k(x, 5) == ("val", "idx")


def test_ctc_loss(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value="ctc")
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert ctc_loss(x, x, x, x) == "ctc"


def test_corrcoef(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value="corrcoef")
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))

    # Eager mode
    config.eager_mode = True
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats.get_active_backend").return_value.execute_op.return_value = mocker.Mock(shape=())
    assert corrcoef(x).config.shape == ()

    # Tracing mode
    config.eager_mode = False
    assert corrcoef(x, x) == "corrcoef"


def test_correlate(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value="correlate")
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))

    # Eager mode
    config.eager_mode = True
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats.get_active_backend").return_value.execute_op.return_value = mocker.Mock(shape=())
    assert correlate(x, x).config.shape == ()

    # Tracing mode
    config.eager_mode = False
    assert correlate(x, x) == "correlate"


def test_cov(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value="cov")
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))

    # Invalid kwargs
    with pytest.raises(ValueError):
        cov(x, invalid=1)

    # Eager mode
    config.eager_mode = True
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats.get_active_backend").return_value.execute_op.return_value = mocker.Mock(shape=())
    assert cov(x).config.shape == ()

    # Tracing mode
    config.eager_mode = False
    assert cov(x, x) == "cov"


def test_cov_with_kwargs(mocker):
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value="cov")
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    config.eager_mode = False
    assert cov(x, rowvar=False) == "cov"
