"""Compiler transformation pass stripping offline diagnostic and symbolic inspection nodes."""

from __future__ import annotations

import logging
import typing
from collections.abc import Mapping

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

OFFLINE_DIAGNOSTIC_OPS: frozenset[str] = frozenset(
    {
        "ExportToDot",
        "PrintGraph",
        "DumpGraph",
        "InspectNode",
        "DotExport",
        "DiagnosticProbe",
    }
)


@typing.runtime_checkable
class _GraphNodesProtocol(typing.Protocol):
    """Protocol for graphs exposing a `nodes` mapping."""

    nodes: Mapping[str, IRNode]


@typing.runtime_checkable
class _GraphOutputsProtocol(typing.Protocol):
    """Protocol for graphs exposing an `outputs` list."""

    outputs: list[str]


@typing.runtime_checkable
class _SortedNodesGraphProtocol(typing.Protocol):
    """Protocol for graphs exposing a mutable `sorted_nodes` list."""

    sorted_nodes: list[IRNode]


def _is_sorted_nodes_holder(val: IRGraph) -> typing.TypeGuard[_SortedNodesGraphProtocol]:
    """Type guard narrowing `val` to `_SortedNodesGraphProtocol` for attribute assignment.

    Args:
        val (IRGraph): Graph instance.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def export_graph_to_dot_string(graph: IRGraph) -> str:
    """Generate a Graphviz DOT representation of an IRGraph.

    Args:
        graph (IRGraph): The computation graph to render.

    Returns:
        str: Graphviz DOT syntax string.
    """
    lines: list[str] = ["digraph G {"]
    lines.append("  rankdir=LR;")
    lines.append('  node [shape=box, fontname="Courier"];')

    raw_graph: TensorOrScalar = graph
    graph_nodes: Mapping[str, IRNode] = raw_graph.nodes if isinstance(raw_graph, _GraphNodesProtocol) else {}
    for node_id, node in graph_nodes.items():
        op = node.op_type if node.op_type else "Unknown"
        label_text = f"{node_id}\\n({op})"
        lines.append(f'  "{node_id}" [label="{label_text}"];')
        for inp in node.inputs:
            lines.append(f'  "{inp}" -> "{node_id}";')

    lines.append("}")
    return "\n".join(lines)


def _find_offline_nodes_and_rewires(graph: IRGraph) -> tuple[list[str], dict[str, str]]:
    """Identify offline diagnostic nodes and map consumer bypass connections.

    Args:
        graph (IRGraph): Target computational graph.

    Returns:
        tuple[list[str], dict[str, str]]: List of node IDs to remove, and dictionary mapping
            removed node IDs to bypass inputs.
    """
    nodes_to_remove: list[str] = []
    rewire_map: dict[str, str] = {}

    for node_id, node in list(graph.nodes.items()):
        op_type = node.op_type
        if op_type not in OFFLINE_DIAGNOSTIC_OPS:
            continue
        nodes_to_remove.append(node_id)
        inputs: list[str] = node.inputs
        primary_input = inputs[0] if inputs else ""
        if primary_input:
            rewire_map[node_id] = primary_input

        attrs = node.attributes
        if op_type == "ExportToDot" and attrs.get("execute_offline", False):
            attrs["dot_output"] = export_graph_to_dot_string(graph)

    # Resolve transitive bypass chains
    for orig, target in list(rewire_map.items()):
        curr = target
        while curr in rewire_map:
            curr = rewire_map[curr]
        rewire_map[orig] = curr

    return nodes_to_remove, rewire_map


def _rewire_graph_edges_and_outputs(
    graph: IRGraph,
    rewire_map: dict[str, str],
    nodes_to_remove: list[str],
) -> None:
    """Rewire remaining node inputs and graph outputs bypassing removed nodes.

    Args:
        graph (IRGraph): Target computational graph.
        rewire_map (dict[str, str]): Mapping of removed nodes to input bypasses.
        nodes_to_remove (list[str]): Node IDs marked for removal.
    """
    for node in graph.nodes.values():
        if node.id in nodes_to_remove:
            continue
        node.inputs = [rewire_map.get(inp, inp) for inp in node.inputs]

    if graph.outputs:
        graph.outputs = [rewire_map.get(out_id, out_id) for out_id in graph.outputs if rewire_map.get(out_id, out_id) not in nodes_to_remove]


def strip_offline_diagnostic_nodes_pass(graph: IRGraph) -> IRGraph:
    """Strip or execute offline symbolic inspection and diagnostic nodes prior to code generation.

    Prevents diagnostic and static graph inspection operations (such as ExportToDot, PrintGraph,
    or DiagnosticProbe) from leaking into numerical compiler backends (WASM, C++, etc.).
    Consumer nodes pointing to diagnostic inspection nodes are rewired to the underlying data inputs,
    and output lists are updated.

    Args:
        graph (IRGraph): The target computational graph.

    Returns:
        IRGraph: Sanitized graph with offline diagnostic nodes removed.
    """
    raw_graph: TensorOrScalar = graph
    if not isinstance(raw_graph, _GraphNodesProtocol) or not raw_graph.nodes:
        return graph

    nodes_to_remove, rewire_map = _find_offline_nodes_and_rewires(graph)
    if not nodes_to_remove:
        _typed_ret_2: IRGraph = graph
        return _typed_ret_2

    _rewire_graph_edges_and_outputs(graph, rewire_map, nodes_to_remove)

    for node_id in nodes_to_remove:
        graph.nodes.pop(node_id, None)

    try:
        sorted_list = topological_sort(graph)
        assert _is_sorted_nodes_holder(graph)
        graph.sorted_nodes = sorted_list
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    _typed_ret_1: IRGraph = graph
    return _typed_ret_1
