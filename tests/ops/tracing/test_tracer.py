"""Tests for test_tracer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.tracing.tracer import TracerTape, get_trace_count, increment_trace_count, reset_trace_count

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_tracer_coverage() -> None:
    """Test function test_tracer_coverage."""

    def mock_func() -> None:
        """Test function mock_func."""
        pass

    cast("Callable[..., EagerTensorProtocol]", reset_trace_count)(mock_func)
    assert cast("Callable[..., EagerTensorProtocol]", get_trace_count)(mock_func) == 0
    cast("Callable[..., EagerTensorProtocol]", increment_trace_count)(mock_func)
    assert cast("Callable[..., EagerTensorProtocol]", get_trace_count)(mock_func) == 1
    cast("Callable[..., EagerTensorProtocol]", reset_trace_count)(mock_func)
    cast("Callable[..., EagerTensorProtocol]", reset_trace_count)(mock_func)
    assert cast("Callable[..., EagerTensorProtocol]", get_trace_count)(mock_func) == 0
    cast("Callable[..., EagerTensorProtocol]", reset_trace_count)(mock_func)

    tape = TracerTape()
    tape.start_tracing()
    tape.add_node(type("obj", (object,), {"id": "1"})())
    tape.stop_tracing()
