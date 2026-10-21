"""Tests for test_tracer."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.tracing.tracer import TracerTape, get_trace_count, increment_trace_count, reset_trace_count
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_tracer_coverage() -> None:
    """Test function test_tracer_coverage."""

    def mock_func() -> None:
        """Test function mock_func."""
        pass

    as_callable(reset_trace_count)(mock_func)
    assert as_callable(get_trace_count)(mock_func) == 0
    as_callable(increment_trace_count)(mock_func)
    assert as_callable(get_trace_count)(mock_func) == 1
    as_callable(reset_trace_count)(mock_func)
    as_callable(reset_trace_count)(mock_func)
    assert as_callable(get_trace_count)(mock_func) == 0
    as_callable(reset_trace_count)(mock_func)

    tape = TracerTape()
    tape.start_tracing()
    tape.add_node(type("obj", (object,), {"id": "1"})())
    tape.stop_tracing()
