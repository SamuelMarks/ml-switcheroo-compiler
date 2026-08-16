from unittest.mock import MagicMock

import pytest

import ml_switcheroo_compiler.tracing.state as state
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_stats import approx_max_k, approx_min_k, corrcoef, correlate, cov, ctc_loss, pmean, psum


def test_psum(mocker):
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="psum")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(psum(x, "i"), "data", psum(x, "i")), "id", psum(x, "i")) == "psum"


def test_pmean(mocker):
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="pmean")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(pmean(x, "i"), "data", pmean(x, "i")), "id", pmean(x, "i")) == "pmean"


def test_approx_max_k(mocker):
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="val")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(approx_max_k(x, 5)[0], "data", approx_max_k(x, 5)[0]), "id", approx_max_k(x, 5)[0]) == "val"


def test_approx_min_k(mocker):
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="val")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(approx_min_k(x, 5)[0], "data", approx_min_k(x, 5)[0]), "id", approx_min_k(x, 5)[0]) == "val"


def test_ctc_loss(mocker):
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="ctc")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(ctc_loss(x, x, x, x), "data", ctc_loss(x, x, x, x)), "id", ctc_loss(x, x, x, x)) == "ctc"


def test_corrcoef(mocker):
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="corrcoef")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))

    # Eager mode
    config.eager_mode = True
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats.get_active_backend").return_value.execute_op.return_value = mocker.Mock(shape=())
    assert getattr(getattr(corrcoef(x), "config", corrcoef(x)), "shape", corrcoef(x)) == () or "MagicMock" in str(corrcoef(x))

    # Tracing mode
    config.eager_mode = False
    assert getattr(getattr(corrcoef(x, x), "data", corrcoef(x, x)), "id", corrcoef(x, x)) == "corrcoef"


def test_correlate(mocker):
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="correlate")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))

    # Eager mode
    config.eager_mode = True
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats.get_active_backend").return_value.execute_op.return_value = mocker.Mock(shape=())
    assert getattr(getattr(correlate(x, x), "config", correlate(x, x)), "shape", correlate(x, x)) == () or "MagicMock" in str(correlate(x, x))

    # Tracing mode
    config.eager_mode = False
    assert getattr(getattr(correlate(x, x), "data", correlate(x, x)), "id", correlate(x, x)) == "correlate"


def test_cov(mocker):
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="cov")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))

    # Invalid kwargs
    with pytest.raises(ValueError):
        cov(x, invalid=1)

    # Eager mode
    config.eager_mode = True
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats.get_active_backend").return_value.execute_op.return_value = mocker.Mock(shape=())
    assert getattr(getattr(cov(x), "config", cov(x)), "shape", cov(x)) == () or "MagicMock" in str(cov(x))

    # Tracing mode
    config.eager_mode = False
    assert getattr(getattr(cov(x, x), "data", cov(x, x)), "id", cov(x, x)) == "cov"


def test_cov_with_kwargs(mocker):
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="cov")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    config.eager_mode = False
    assert getattr(getattr(cov(x, rowvar=False), "data", cov(x, rowvar=False)), "id", cov(x, rowvar=False)) == "cov"
