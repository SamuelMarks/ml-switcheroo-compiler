"""Tests for test_frontend_stats."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock

import pytest

import ml_switcheroo_compiler.tracing.state as state
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_stats import approx_max_k, approx_min_k, corrcoef, correlate, cov, ctc_loss, pmean, psum
from tests.testing_utils import ensure_eager_tensor

if typing.TYPE_CHECKING:
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
    assert ensure_eager_tensor(ensure_eager_tensor(psum(x, "i")).data).id == "psum"


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
    assert ensure_eager_tensor(ensure_eager_tensor(pmean(x, "i")).data).id == "pmean"


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
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_eager_tensor(approx_max_k(x, 5))[0]).data).id == "val"


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
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_eager_tensor(approx_min_k(x, 5))[0]).data).id == "val"


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
    assert ensure_eager_tensor(ensure_eager_tensor(ctc_loss(x, x, x, x)).data).id == "ctc"


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
        assert ensure_eager_tensor(res1).shape == (2, 2)
        res2 = corrcoef(x, x)  # y is not None
        assert ensure_eager_tensor(res2).shape == (4, 4)
    finally:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = True
        enable_compile()

    # Tracing mode
    assert ensure_eager_tensor(ensure_eager_tensor(corrcoef(x)).data).id == "corrcoef"
    assert ensure_eager_tensor(ensure_eager_tensor(corrcoef(x, x)).data).id == "corrcoef"


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
        assert ensure_eager_tensor(correlate(x, x)).shape == (1,)
    finally:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = True
        enable_compile()

    # Tracing mode
    assert ensure_eager_tensor(ensure_eager_tensor(correlate(x, x)).data).id == "correlate"


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
        assert ensure_eager_tensor(cov(x)).shape == (2, 2)
        assert ensure_eager_tensor(cov(x, x)).shape == (4, 4)
    finally:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = True
        enable_compile()

    # Tracing mode
    assert ensure_eager_tensor(ensure_eager_tensor(cov(x, x)).data).id == "cov"


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
    assert ensure_eager_tensor(ensure_eager_tensor(cov(x, rowvar=False)).data).id == "cov"
