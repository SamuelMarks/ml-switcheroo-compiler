"""Provides required module functionality."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.tracing.tracer import _tracer


class MockNode(IRNode):
    """Mock Node."""


def test_tracer_coverage_brute() -> None:
    """Execute the requested function."""
    _tracer.start_tracing()
    config.current_stream = "custom"

    n = MockNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
    # The condition is: if hasattr(node, "stream") and node.stream is None
    # and config.current_stream != "default":
    # Let's set hasattr True, and stream is None.
    n.stream = None

    _tracer.add_node(n)

    assert n.stream == "custom"

    _tracer.stop_tracing()
    config.current_stream = "default"
