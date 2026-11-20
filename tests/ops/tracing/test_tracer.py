"""Tests for test_tracer."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.tracing.tracer import TracerTape, get_trace_count, increment_trace_count, reset_trace_count

if typing.TYPE_CHECKING:
    pass


def test_tracer_coverage() -> None:
    """Test function test_tracer_coverage."""

    def mock_func() -> None:
        """Test function mock_func."""
        pass

    reset_trace_count(mock_func)
    assert get_trace_count(mock_func) == 0
    increment_trace_count(mock_func)
    assert get_trace_count(mock_func) == 1
    reset_trace_count(mock_func)
    reset_trace_count(mock_func)
    assert get_trace_count(mock_func) == 0
    reset_trace_count(mock_func)

    from ml_switcheroo_ir import LogicalNode

    tape = TracerTape()
    tape.start_tracing()
    tape.add_node(LogicalNode(id="1", op_type="Identity"))
    tape.stop_tracing()
