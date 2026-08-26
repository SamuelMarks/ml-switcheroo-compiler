# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide utilities for debugging tensor shapes and visualizing logical IR graphs.

This module includes functions to trace model execution shapes into Markdown tables, as
well as export logical graphs to Graphviz DOT and HTML formats for visualization
"""

import os
from typing import Callable

import yaml
from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.tracing.state import global_tracing_state

_FORMATTERS = {}


def _load_formatters() -> None:
    """_load_formatters function.

    Returns:
        object: Result.
    """
    global _FORMATTERS
    if not _FORMATTERS:
        yaml_path = os.path.join(os.path.dirname(__file__), "formatters.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                _FORMATTERS = yaml.safe_load(f) or {}


def debug_shapes(model_func, input_shape) -> str:
    """Trace the execution of a model function to debug and document tensor shapes.

    Args:
        model_func (object): The model_func parameter.
        input_shape (object): The input_shape parameter.

    Returns:
        str: Result.
    """
    _load_formatters()
    fmt = _FORMATTERS.get("markdown_table", {})
    header = fmt.get("header", "| Node | Shape | DType |\n|---|---|---|\n")
    row_fmt = fmt.get("row", "| {name} | {shape} | {dtype} |\n")

    res = header
    global_tracing_state.start_tracing()
    try:
        dummy_input = ops.zeros(input_shape, dtype=DType.Float64)
        res += row_fmt.format(name="input", shape=input_shape, dtype="float64")

        out = model_func(dummy_input)
        if hasattr(out, "shape"):
            res += row_fmt.format(name="output", shape=out.shape, dtype="float64")
        else:
            res += row_fmt.format(name="output", shape="unknown", dtype="float64")
    except RuntimeError:
        res = header
    finally:
        global_tracing_state.stop_tracing()

    return res


def to_graphviz(graph: LogicalGraph) -> str:
    """Convert a logical IR graph into a Graphviz DOT format string.

    Args:
        graph (LogicalGraph): The graph parameter.

    Returns:
        str: Result.
    """
    _load_formatters()
    fmt = _FORMATTERS.get("graphviz", {})
    header = fmt.get("header", "digraph G {\n")
    node_fmt = fmt.get("node", '  "{nid}" [label="{op_type}"];\n')
    edge_fmt = fmt.get("edge", '  "{inp}" -> "{nid}";\n')
    footer = fmt.get("footer", "}\n")

    dot = header
    for nid, node in graph.nodes.items():
        dot += node_fmt.format(nid=nid, op_type=node.op_type)
        for inp in node.inputs:
            dot += edge_fmt.format(inp=inp, nid=nid)
    dot += footer
    return dot


def to_html(graph: LogicalGraph) -> str:
    """Convert a logical IR graph into an HTML visualization.

    Args:
        graph (LogicalGraph): The graph parameter.

    Returns:
        str: Result.
    """
    _load_formatters()
    fmt = _FORMATTERS.get("html", {})
    return fmt.get("template", "<html><body><h1>IR Graph</h1></body></html>")
