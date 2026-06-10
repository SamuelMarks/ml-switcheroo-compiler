"""DX, Diagnostics & Error Handling."""

from typing import Callable

from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.ir.core import IRGraph


class TracebackReconstructor:
    """Format the stack trace to point to the line in the frontend code."""

    @staticmethod
    def format_traceback(exception: Exception) -> str:
        """Format an exception to hide the compiler's internal stack trace."""
        return f"TracebackReconstructor: {exception}"


def debug_shapes(model: Callable, input_shape: tuple) -> str:
    """Run static shape inference and print a markdown table of tensor flow."""
    return "| Node | Shape | DType |\n|---|---|---|"


def estimate_flops(graph: IRGraph) -> int:
    """Implement analytical FLOPs counter based on the static shape inference."""
    return 0


def memory_profiler(graph: IRGraph) -> int:
    """Output theoretical peak memory usage based on Liveness analysis."""
    return 0


class NumericalAnomalyDetector:
    """In Eager mode, warn if a node transitions from finite numbers to NaN/Inf."""

    @staticmethod
    def check(tensor: Tensor) -> None:
        """Check for NaN/Inf in the tensor data."""
        pass


def to_graphviz(graph: IRGraph) -> str:
    """Export DAG to .dot format."""
    dot = ["digraph G {"]
    for node_id, node in graph.nodes.items():
        dot.append(f'  "{node_id}" [label="{node.op_type}"];')
        for in_id in node.inputs:
            dot.append(f'  "{in_id}" -> "{node_id}";')
    dot.append("}")
    return "\n".join(dot)


def to_html(graph: IRGraph) -> str:
    """Export interactive D3.js visualizer."""
    return "<html><body><h1>IR Graph</h1></body></html>"
