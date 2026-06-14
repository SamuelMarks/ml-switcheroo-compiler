"""Dead Code Elimination pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def dce_pass(graph: IRGraph) -> bool:
    """In-place Dead Code Elimination (DCE).

    Removes nodes that do not contribute to the graph outputs

    graph (IRGraph): The input graph

    Returns:
    bool: True if the graph was modified

    Args:
        graph (IRGraph): Argument graph
    """
    reachable: set[str] = set(graph.outputs)

    # We must explicitly check nodes to build reachability correctly
    # We trace backwards from the outputs
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    for node in reversed(sorted_nodes):
        if node.id in reachable:
            for inp in node.inputs:
                reachable.add(inp)

    nodes_to_remove = [nid for nid in graph.nodes if nid not in reachable]

    for nid in nodes_to_remove:
        del graph.nodes[nid]

    return len(nodes_to_remove) > 0
