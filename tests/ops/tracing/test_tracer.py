"""Tests for test_tracer."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.tracing.tracer import TracerTape, get_trace_count, increment_trace_count, reset_trace_count
from tests.testing_utils import ensure_callable

if typing.TYPE_CHECKING:
    pass


def test_tracer_coverage() -> None:
    """Test function test_tracer_coverage."""

    def mock_func() -> None:
        """Test function mock_func."""
        pass

    ensure_callable(reset_trace_count)(mock_func)
    assert ensure_callable(get_trace_count)(mock_func) == 0
    ensure_callable(increment_trace_count)(mock_func)
    assert ensure_callable(get_trace_count)(mock_func) == 1
    ensure_callable(reset_trace_count)(mock_func)
    ensure_callable(reset_trace_count)(mock_func)
    assert ensure_callable(get_trace_count)(mock_func) == 0
    ensure_callable(reset_trace_count)(mock_func)

    tape = TracerTape()
    tape.start_tracing()
    tape.add_node(type("obj", (), {"id": "1"})())
    tape.stop_tracing()
