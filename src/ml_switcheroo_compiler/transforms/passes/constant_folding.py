"""Constant Folding pass."""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _are_all_inputs_constant(canonical_inputs: list[str], graph: IRGraph) -> bool:
    """Execute _are_all_inputs_constant.

    Args:
        canonical_inputs (Any): Argument canonical_inputs.
        graph (Any): Argument graph.

    Returns:
    Any: The result.
    """
    if not canonical_inputs:
        return False
    for inp in canonical_inputs:
        if inp not in graph.nodes or graph.nodes[inp].op_type != "Constant":
            return False
    return True


def _evaluate_constant_node(
    node: object, canonical_inputs: list[str], graph: IRGraph, backend: object
) -> object:
    """Execute _evaluate_constant_node.

    Args:
        node (Any): Argument node.
        canonical_inputs (Any): Argument canonical_inputs.
        graph (Any): Argument graph.
        backend (Any): Argument backend.

    Returns:
    Any: The result.
    """
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.interpreter import evaluate_graph

    subgraph = LogicalGraph(outputs=[node.id])
    for inp in canonical_inputs:
        subgraph.nodes[inp] = graph.nodes[inp]
    subgraph.nodes[node.id] = LogicalNode(
        id=node.id,
        op_type=node.op_type,
        attributes=dict(node.attributes),
        inputs=list(canonical_inputs),
        shape_metadata=node.shape_metadata,
    )
    outputs = evaluate_graph(subgraph, {})
    val = outputs[node.id]

    if (hasattr(val, "size") and val.size == 1) or (hasattr(val, "numel") and val.numel() == 1):
        val = backend.item(val)
    return val


def constant_folding_pass(graph: IRGraph) -> bool:
    """In-place Constant Folding.

    Evaluates pure operations on constant inputs eagerly using the interpreter

    Args:
        graph (IRGraph): Argument graph

    Returns:
    bool: True if the graph was modified
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    id_map: dict[str, str] = {}
    backend = get_active_backend()

    for node in sorted_nodes:
        canonical_inputs = [id_map.get(inp, inp) for inp in node.inputs]

        if _are_all_inputs_constant(canonical_inputs, graph):
            try:
                val = _evaluate_constant_node(node, canonical_inputs, graph, backend)
                graph.nodes[node.id].op_type = "Constant"
                graph.nodes[node.id].attributes = {"value": val}
                graph.nodes[node.id].inputs = []
                id_map[node.id] = node.id
                modified = True
                continue
            except (ValueError, TypeError, NotImplementedError, RuntimeError):
                pass

        id_map[node.id] = node.id

    return modified
