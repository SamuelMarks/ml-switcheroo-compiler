"""Unit tests for the unified operation dispatcher (`ops/dispatcher.py`)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.dispatcher import (
    dispatch_backend_eager_op,
    dispatch_multi_op,
    dispatch_op,
    dispatch_submodule_op,
    is_eager_execution_active,
)
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_dispatch_op_eager_single_and_tuples() -> None:
    """Verify `dispatch_op` handles single tensors, 1-tuples, and multi-output tuples in eager mode."""
    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        assert is_eager_execution_active() is True
        with patch.object(EagerEvaluator, "evaluate", return_value="evaluated_val") as mock_eval:
            result = dispatch_op("Add", 1, 2, keyword="test")
            assert result == "evaluated_val"
            mock_eval.assert_called_once_with("Add", 1, 2, keyword="test")

        with patch.object(EagerEvaluator, "evaluate", return_value="single_multi"):
            assert dispatch_multi_op("Add", 1, 2) == "single_multi"

        with patch.object(EagerEvaluator, "evaluate", return_value=("only_one",)):
            assert dispatch_op("Identity", 1) == "only_one"
            assert dispatch_multi_op("Identity", 1) == "only_one"

        with patch.object(EagerEvaluator, "evaluate", return_value=("u", "s", "vt")):
            assert dispatch_multi_op("Svd", 1) == ("u", "s", "vt")
    finally:
        config.eager_mode = original_eager


def test_dispatch_op_tracing_disabled_raises() -> None:
    """Verify `dispatch_op` raises RuntimeError when eager_mode is False and not tracing."""
    original_eager = config.eager_mode
    original_tracing = global_tracing_state.is_tracing
    try:
        config.eager_mode = False
        global_tracing_state.is_tracing = False
        with pytest.raises(RuntimeError, match="Cannot emit Add node outside of a tracing context"):
            dispatch_op("Add", 1, 2)
    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = original_tracing


def test_dispatch_op_tracing_active() -> None:
    """Verify `dispatch_op` routes to `TracingNodeBuilder` when tracing is active."""
    original_eager = config.eager_mode
    original_tracing = global_tracing_state.is_tracing
    try:
        config.eager_mode = False
        global_tracing_state.is_tracing = True
        with patch.object(TracingNodeBuilder, "emit_tracing_node", return_value="traced_node") as mock_emit:
            result = dispatch_op("Mul", 3, 4, attr="val")
            assert result == "traced_node"
            mock_emit.assert_called_once_with("Mul", 3, 4, attr="val")
            assert dispatch_multi_op("Svd", 3) == "traced_node"
    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = original_tracing


def test_dispatch_backend_eager_op_and_submodule_op() -> None:
    """Verify `dispatch_backend_eager_op` and `dispatch_submodule_op` across single/tuple and eager/tracing."""
    t_in = Tensor(np.array([1.0, 2.0], dtype=np.float32), TensorConfig((2,), DType.Float32, "cpu"))
    single_out = dispatch_backend_eager_op("Add", (t_in, t_in), {})
    assert isinstance(single_out, Tensor)

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = (np.array([1.0]), np.array([2.0]))
    mock_backend.array.side_effect = lambda d: np.asarray(d)
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
        tup_out = dispatch_backend_eager_op("Split", (1.0,), {})
        assert isinstance(tup_out, tuple) and len(tup_out) == 2

    original_eager = config.eager_mode
    try:
        config.eager_mode = True

        class _WithModuleBackend:
            """Backend stub exposing `.module = np`."""

            module = np

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=_WithModuleBackend()):
            svd_res = dispatch_submodule_op("linalg", "svd", (np.eye(2, dtype=np.float32),), {})
            assert svd_res is not None

        def _provide_adapter() -> OpValue:
            """Return an adapted NumPy module without `.module` attribute.

            Returns:
                OpValue: Adapted backend callable/module wrapper.
            """
            from ml_switcheroo_compiler.backends.protocols import StaticEagerBackendAdapter

            return StaticEagerBackendAdapter(backend_name="np_sub", submodules={"linalg": np.linalg}).get_op

        with pytest.raises(ValueError, match="not supported"):
            dispatch_submodule_op(
                "linalg",
                "svd",
                (np.eye(2, dtype=np.float32),),
                {},
                backend_getter=_provide_adapter,
            )

        config.eager_mode = False
        g = global_tracing_state.start_tracing("SubmodTrace")
        try:
            traced_dt = dispatch_submodule_op("linalg", "UnknownSubmodOpXyz", (t_in,), {})
            assert traced_dt is not None
            traced_empty = dispatch_submodule_op("linalg", "UnknownSubmodOpXyz", (), {})
            assert traced_empty is not None
        finally:
            assert global_tracing_state.stop_tracing() is g
    finally:
        config.eager_mode = original_eager
