"""Provides required module functionality."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.tracing.state import global_tracing_state


class MockNode(IRNode):
    """Mock Node."""


def testglobal_tracing_state_coverage_brute() -> None:
    """Execute the requested function."""
    global_tracing_state.start_tracing()
    config.current_stream = "custom"

    n = MockNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
    # The condition is: if hasattr(node, "stream") and node.stream is None
    # and config.current_stream != "default":
    # Let's set hasattr True, and stream is None.
    n.stream = None

    global_tracing_state.add_node(n)

    assert n.stream == "custom"

    global_tracing_state.stop_tracing()
    config.current_stream = "default"
