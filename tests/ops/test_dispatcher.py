"""Tests for operation dispatcher."""

from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_dispatch_op_eager_mode() -> None:
    """Test dispatch_op routes to EagerEvaluator when eager_mode is True.

    Returns:
        None.
    """
    original_eager = config.eager_mode
    try:
        config.eager_mode = True
        with patch.object(EagerEvaluator, "evaluate", return_value="evaluated_val") as mock_eval:
            result = dispatch_op("Add", 1, 2, keyword="test")
            assert result == "evaluated_val"
            mock_eval.assert_called_once_with("Add", 1, 2, keyword="test")
    finally:
        config.eager_mode = original_eager


def test_dispatch_op_tracing_disabled_raises() -> None:
    """Test dispatch_op raises RuntimeError when eager_mode is False and not tracing.

    Returns:
        None.
    """
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
    """Test dispatch_op routes to TracingNodeBuilder when eager_mode is False and tracing is active.

    Returns:
        None.
    """
    original_eager = config.eager_mode
    original_tracing = global_tracing_state.is_tracing
    try:
        config.eager_mode = False
        global_tracing_state.is_tracing = True
        with patch.object(TracingNodeBuilder, "emit_tracing_node", return_value="traced_node") as mock_emit:
            result = dispatch_op("Mul", 3, 4, attr="val")
            assert result == "traced_node"
            mock_emit.assert_called_once_with("Mul", 3, 4, attr="val")
    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = original_tracing
