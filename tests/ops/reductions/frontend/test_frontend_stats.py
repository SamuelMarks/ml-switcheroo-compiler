"""Tests for test_frontend_stats."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import ml_switcheroo_compiler.tracing.state as state
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.api_stats import (
    approx_max_k,
    approx_min_k,
    corrcoef,
    correlate,
    cov,
    ctc_loss,
    pmean,
    psum,
)
from tests.testing_utils import adapt_backend_module, coerce_operand_list, extract_test_data, extract_test_shape

if typing.TYPE_CHECKING:
    pass


def test_psum(mocker: MockerFixture) -> None:
    """Test function test_psum.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch(
        "ml_switcheroo_compiler.ops.reductions.api_stats._emit_reduction_node",
        return_value=MagicMock(data=MagicMock(id="psum")),
    )
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert adapt_backend_module(extract_test_data(psum(x, "i"))).require_symbol("id") == "psum"


def test_pmean(mocker: MockerFixture) -> None:
    """Test function test_pmean.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch(
        "ml_switcheroo_compiler.ops.reductions.api_stats._emit_reduction_node",
        return_value=MagicMock(data=MagicMock(id="pmean")),
    )
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert adapt_backend_module(extract_test_data(pmean(x, "i"))).require_symbol("id") == "pmean"


def test_approx_max_k(mocker: MockerFixture) -> None:
    """Test function test_approx_max_k.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch(
        "ml_switcheroo_compiler.ops.reductions.api_stats._emit_reduction_node",
        return_value=MagicMock(data=MagicMock(id="val")),
    )
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert (
        adapt_backend_module(extract_test_data(coerce_operand_list(approx_max_k(x, 5))[0])).require_symbol("id")
        == "val"
    )


def test_approx_min_k(mocker: MockerFixture) -> None:
    """Test function test_approx_min_k.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch(
        "ml_switcheroo_compiler.ops.reductions.api_stats._emit_reduction_node",
        return_value=MagicMock(data=MagicMock(id="val")),
    )
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert (
        adapt_backend_module(extract_test_data(coerce_operand_list(approx_min_k(x, 5))[0])).require_symbol("id")
        == "val"
    )


def test_ctc_loss(mocker: MockerFixture) -> None:
    """Test function test_ctc_loss.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch(
        "ml_switcheroo_compiler.ops.reductions.api_stats._emit_reduction_node",
        return_value=MagicMock(data=MagicMock(id="ctc")),
    )
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    assert adapt_backend_module(ctc_loss(x, x, x, x).data).require_symbol("id") == "ctc"


def test_corrcoef(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_corrcoef.

    Args:
        mocker: Test parameter `mocker`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    import sys

    from ml_switcheroo_compiler.core.config import disable_compile, enable_compile

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch(
        "ml_switcheroo_compiler.ops.reductions.api_stats._emit_reduction_node",
        return_value=MagicMock(data=MagicMock(id="corrcoef")),
    )
    import numpy as np

    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))

    # Eager mode
    try:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = False
        disable_compile()  # sets eager_mode = True

        res1 = corrcoef(x)  # y is None
        assert extract_test_shape(res1) == (2, 2)
        res2 = corrcoef(x, x)  # y is not None
        assert extract_test_shape(res2) == (4, 4)
    finally:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = True
        enable_compile()

    # Tracing mode
    assert adapt_backend_module(extract_test_data(corrcoef(x))).require_symbol("id") == "corrcoef"
    assert adapt_backend_module(extract_test_data(corrcoef(x, x))).require_symbol("id") == "corrcoef"


def test_correlate(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_correlate.

    Args:
        mocker: Test parameter `mocker`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    import sys

    from ml_switcheroo_compiler.core.config import disable_compile, enable_compile

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch(
        "ml_switcheroo_compiler.ops.reductions.api_stats._emit_reduction_node",
        return_value=MagicMock(data=MagicMock(id="correlate")),
    )
    import numpy as np

    x = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))

    # Eager mode
    try:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = False
        disable_compile()
        assert extract_test_shape(correlate(x, x)) == (1,)
    finally:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = True
        enable_compile()

    # Tracing mode
    assert adapt_backend_module(extract_test_data(correlate(x, x))).require_symbol("id") == "correlate"


def test_cov(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_cov.

    Args:
        mocker: Test parameter `mocker`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    import sys

    from ml_switcheroo_compiler.core.config import disable_compile, enable_compile

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch(
        "ml_switcheroo_compiler.ops.reductions.api_stats._emit_reduction_node",
        return_value=MagicMock(data=MagicMock(id="cov")),
    )
    import numpy as np

    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))

    # Invalid kwargs
    with pytest.raises(ValueError):
        cov(x, invalid=1)

    # Eager mode
    try:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = False
        disable_compile()
        assert extract_test_shape(cov(x)) == (2, 2)
        assert extract_test_shape(cov(x, x)) == (4, 4)
    finally:
        sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state.is_tracing = True
        enable_compile()

    # Tracing mode
    assert adapt_backend_module(extract_test_data(cov(x, x))).require_symbol("id") == "cov"


def test_cov_with_kwargs(mocker: MockerFixture) -> None:
    """Test function test_cov_with_kwargs.

    Args:
        mocker: Test parameter `mocker`.
    """
    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = MagicMock()
    mocker.patch(
        "ml_switcheroo_compiler.ops.reductions.api_stats._emit_reduction_node",
        return_value=MagicMock(data=MagicMock(id="cov")),
    )
    x = Tensor(1.0, TensorConfig((), "float32", "cpu"))
    config.eager_mode = False
    assert adapt_backend_module(extract_test_data(cov(x, rowvar=False))).require_symbol("id") == "cov"
