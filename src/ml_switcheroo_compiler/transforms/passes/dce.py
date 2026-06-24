"""Dead Code Elimination pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _find_side_effect_nodes(graph: IRGraph) -> set[str]:
    """Function docstring.

    Args:
        graph: Arg.
    """
    side_effect_ops = {"Assert", "AssignVariable", "Print"}
    return {node.id for node in graph.nodes.values() if node.op_type in side_effect_ops}


def _build_reachable_set(graph: IRGraph, initial_reachable: set[str]) -> set[str]:
    """Function docstring.

    Args:
        graph: Arg.
        initial_reachable: Arg.
    """
    reachable = set(initial_reachable)
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    for node in reversed(sorted_nodes):
        if node.id in reachable:
            for inp in node.inputs:
                reachable.add(inp)
    return reachable


def dce_pass(graph: IRGraph) -> bool:
    """In-place Dead Code Elimination (DCE).

    Removes nodes that do not contribute to the graph outputs

    graph (IRGraph): The input graph

    Returns:
    bool: True if the graph was modified

    Args:
        graph (IRGraph): Argument graph
    """
    initial_reachable = set(graph.outputs) | _find_side_effect_nodes(graph)
    reachable = _build_reachable_set(graph, initial_reachable)

    nodes_to_remove = []
    for nid in graph.nodes:
        if nid not in reachable:
            nodes_to_remove.append(nid)

    for nid in nodes_to_remove:
        del graph.nodes[nid]

    return len(nodes_to_remove) > 0
