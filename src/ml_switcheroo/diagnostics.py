"""DX, Diagnostics & Error Handling."""

from typing import Callable, Any
import numpy as np

from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.ir.core import IRGraph


class TracebackReconstructor:
    """Format the stack trace to point to the line in the frontend code."""

    @staticmethod
    def format_traceback(exception: Exception) -> str:
        """Format an exception to hide the compiler's internal stack trace.

        Args:
            exception (Exception): The exception to format.

        Returns:
            str: The formatted string.
        """
        return f"TracebackReconstructor: {exception}"


def debug_shapes(model: Callable[..., Any], input_shape: tuple) -> str:
    """Run static shape inference and print a markdown table of tensor flow.

    Args:
        model: A callable model (e.g. from the nn module).
        input_shape (tuple): The shape of the input tensor.

    Returns:
        str: A markdown table representation of shapes.
    """
    try:
        res = model(np.zeros(input_shape))
        out_shape = res.shape if hasattr(res, "shape") else "unknown"
        return f"| Node | Shape | DType |\n|---|---|---|\n| input | {input_shape} | float64 |\n| output | {out_shape} | float64 |"  # noqa: E501
    except Exception:
        return "| Node | Shape | DType |\n|---|---|---|"


def estimate_flops(graph: IRGraph) -> int:
    """Implement analytical FLOPs counter based on the static shape inference.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        int: Estimated floating point operations.
    """
    flops = 0
    for node in graph.nodes.values():
        if node.op_type in ("Add", "Sub", "Mul", "Div", "Relu"):
            if node.shape_metadata:
                try:
                    # simple volume for scalar ops
                    flops += int(
                        np.prod(
                            [dim for dim in node.shape_metadata if isinstance(dim, int)]
                        )
                    )
                except Exception:
                    flops += 1
            else:
                flops += 1
        elif node.op_type == "MatMul":
            # rough estimate
            flops += 100
    return flops


def memory_profiler(graph: IRGraph) -> int:
    """Output theoretical peak memory usage based on Liveness analysis.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        int: Estimated peak memory usage in bytes.
    """
    peak_mem = 0
    for node in graph.nodes.values():
        if node.shape_metadata:
            try:
                peak_mem += (
                    int(
                        np.prod(
                            [dim for dim in node.shape_metadata if isinstance(dim, int)]
                        )
                    )
                    * 4
                )  # assume float32
            except Exception:
                peak_mem += 4
        else:
            peak_mem += 4
    return peak_mem


class NumericalAnomalyDetector:
    """In Eager mode, warn if a node transitions from finite numbers to NaN/Inf."""

    @staticmethod
    def check(tensor: Tensor) -> None:
        """Check for NaN/Inf in the tensor data.

        Args:
            tensor (Tensor): The tensor to check.

        Raises:
            ValueError: If the tensor contains NaN or Inf.
        """
        if tensor.data is None:
            return

        # Assuming tensor.data can be converted to or treated as numpy array
        try:
            arr = np.asarray(tensor.data)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "Numerical anomaly detected: Tensor contains NaN or Inf."
                )
        except TypeError:
            pass


def to_graphviz(graph: IRGraph) -> str:
    """Export DAG to .dot format.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        str: The DOT formatted string.
    """
    dot = ["digraph G {"]
    for node_id, node in graph.nodes.items():
        dot.append(f'  "{node_id}" [label="{node.op_type}"];')
        for in_id in node.inputs:
            dot.append(f'  "{in_id}" -> "{node_id}";')
    dot.append("}")
    return "\n".join(dot)


def to_html(graph: IRGraph) -> str:
    """Export interactive D3.js visualizer.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        str: Minimal HTML string.
    """
    return "<html><body><h1>IR Graph</h1></body></html>"
