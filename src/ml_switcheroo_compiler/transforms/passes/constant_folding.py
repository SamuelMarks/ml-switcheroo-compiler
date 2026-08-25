# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Constant Folding pass."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.interpreter import evaluate_graph
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _are_all_inputs_constant(canonical_inputs: list[str], graph: IRGraph) -> bool:
    """Evaluate _are_all_inputs_constant operation.

    Args:
        canonical_inputs (list): The canonical_inputs parameter.
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    if not canonical_inputs:
        return False
    for inp in canonical_inputs:
        if inp not in graph.nodes or graph.nodes[inp].op_type != "Constant":
            return False
    return True


def _evaluate_constant_node(node: object, canonical_inputs: list[str], graph: IRGraph, backend: object) -> object:
    """Evaluate _evaluate_constant_node operation.

    Args:
        node (object): The node parameter.
        canonical_inputs (list): The canonical_inputs parameter.
        graph (IRGraph): The graph parameter.
        backend (object): The backend parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    subgraph: object = LogicalGraph(outputs=[node.id])
    for inp in canonical_inputs:
        subgraph.nodes[inp] = graph.nodes[inp]
    subgraph.nodes[node.id] = LogicalNode(
        id=node.id,
        op_type=node.op_type,
        attributes=dict(node.attributes),
        inputs=list(canonical_inputs),
        shape_metadata=node.shape_metadata,
    )
    outputs: object = evaluate_graph(subgraph, {})
    val: object = outputs[node.id]
    if (hasattr(val, "size") and val.size == 1) or (hasattr(val, "numel") and val.numel() == 1):
        val: object = backend.item(val)
    return val


def constant_folding_pass(graph: IRGraph) -> bool:
    """In-place Constant Folding.

    Raises:
        Exception: An exception.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    modified: object = False
    sorted_nodes: object = DAGTopologicalSorter.sort(graph)
    id_map: dict[str, str] = {}
    backend: object = get_active_backend()
    for node in sorted_nodes:
        canonical_inputs: object = [id_map.get(inp, inp) for inp in node.inputs]
        if _are_all_inputs_constant(canonical_inputs, graph):
            try:
                val: object = _evaluate_constant_node(node, canonical_inputs, graph, backend)
                graph.nodes[node.id].op_type = "Constant"
                graph.nodes[node.id].attributes = {"value": val}
                graph.nodes[node.id].inputs = []
                id_map[node.id] = node.id
                modified: object = True
                continue
            except (ValueError, TypeError, RuntimeError) as e:
                import logging

                logging.getLogger(__name__).debug(f"Failed to constant fold {node.op_type}: {e}")
                continue
            except Exception as e:
                if type(e).__name__ == "UnimplementedMathError":
                    import logging

                    logging.getLogger(__name__).debug(f"Unimplemented math for constant folding {node.op_type}: {e}")
                    continue
                else:
                    raise
        id_map[node.id] = node.id
    return modified
