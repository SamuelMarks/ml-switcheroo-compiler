"""Tests for test_frontend_stats."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

import ml_switcheroo_compiler.tracing.state as state
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_stats import approx_max_k, approx_min_k, corrcoef, correlate, cov, ctc_loss, pmean, psum

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_psum(mocker: EagerTensorProtocol) -> None:
    """Test function test_psum.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="psum")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(psum(x, "i"), "data", psum(x, "i")), "id", psum(x, "i")) == "psum"


def test_pmean(mocker: EagerTensorProtocol) -> None:
    """Test function test_pmean.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="pmean")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(pmean(x, "i"), "data", pmean(x, "i")), "id", pmean(x, "i")) == "pmean"


def test_approx_max_k(mocker: EagerTensorProtocol) -> None:
    """Test function test_approx_max_k.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="val")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(cast("EagerTensorProtocol", getattr(approx_max_k(x, 5), "__getitem__")(0)), "data", cast("EagerTensorProtocol", getattr(approx_max_k(x, 5), "__getitem__")(0))), "id", cast("EagerTensorProtocol", getattr(approx_max_k(x, 5), "__getitem__")(0))) == "val"


def test_approx_min_k(mocker: EagerTensorProtocol) -> None:
    """Test function test_approx_min_k.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="val")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(cast("EagerTensorProtocol", getattr(approx_min_k(x, 5), "__getitem__")(0)), "data", cast("EagerTensorProtocol", getattr(approx_min_k(x, 5), "__getitem__")(0))), "id", cast("EagerTensorProtocol", getattr(approx_min_k(x, 5), "__getitem__")(0))) == "val"


def test_ctc_loss(mocker: EagerTensorProtocol) -> None:
    """Test function test_ctc_loss.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="ctc")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert getattr(getattr(ctc_loss(x, x, x, x), "data", ctc_loss(x, x, x, x)), "id", ctc_loss(x, x, x, x)) == "ctc"


def test_corrcoef(mocker: EagerTensorProtocol, monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_corrcoef.

    Args:
        mocker: Test parameter `mocker`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    import sys

    from ml_switcheroo_compiler.core.config import disable_compile, enable_compile

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="corrcoef")))
    import numpy as np

    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))

    # Eager mode
    try:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = False
        disable_compile()  # sets eager_mode = True

        res1 = corrcoef(x)  # y is None
        assert cast("EagerTensorProtocol", getattr(res1, "shape")) == (2, 2)
        res2 = corrcoef(x, x)  # y is not None
        assert cast("EagerTensorProtocol", getattr(res2, "shape")) == (4, 4)
    finally:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = True
        enable_compile()

    # Tracing mode
    assert getattr(getattr(corrcoef(x), "data", corrcoef(x)), "id", corrcoef(x)) == "corrcoef"
    assert getattr(getattr(corrcoef(x, x), "data", corrcoef(x, x)), "id", corrcoef(x, x)) == "corrcoef"


def test_correlate(mocker: EagerTensorProtocol, monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_correlate.

    Args:
        mocker: Test parameter `mocker`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    import sys

    from ml_switcheroo_compiler.core.config import disable_compile, enable_compile

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="correlate")))
    import numpy as np

    x = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))

    # Eager mode
    try:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = False
        disable_compile()
        assert cast("EagerTensorProtocol", getattr(correlate(x, x), "shape")) == (1,)
    finally:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = True
        enable_compile()

    # Tracing mode
    assert getattr(getattr(correlate(x, x), "data", correlate(x, x)), "id", correlate(x, x)) == "correlate"


def test_cov(mocker: EagerTensorProtocol, monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_cov.

    Args:
        mocker: Test parameter `mocker`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    import sys

    from ml_switcheroo_compiler.core.config import disable_compile, enable_compile

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="cov")))
    import numpy as np

    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))

    # Invalid kwargs
    with pytest.raises(ValueError):
        cov(x, invalid=1)

    # Eager mode
    try:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = False
        disable_compile()
        assert cast("EagerTensorProtocol", getattr(cov(x), "shape")) == (2, 2)
        assert cast("EagerTensorProtocol", getattr(cov(x, x), "shape")) == (4, 4)
    finally:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = True
        enable_compile()

    # Tracing mode
    assert getattr(getattr(cov(x, x), "data", cov(x, x)), "id", cov(x, x)) == "cov"


def test_cov_with_kwargs(mocker: EagerTensorProtocol) -> None:
    """Test function test_cov_with_kwargs.

    Args:
        mocker: Test parameter `mocker`.
    """
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_stats._emit_reduction_node", return_value=MagicMock(data=MagicMock(id="cov")))
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    config.eager_mode = False
    assert getattr(getattr(cov(x, rowvar=False), "data", cov(x, rowvar=False)), "id", cov(x, rowvar=False)) == "cov"
