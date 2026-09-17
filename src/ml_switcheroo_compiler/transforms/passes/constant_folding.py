# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Constant Folding pass evaluated via reference eager backend."""

from typing import Any

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import BackendRegistry, get_active_backend
from ml_switcheroo_compiler.interpreter import evaluate_graph
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _are_all_inputs_constant(canonical_inputs: list[str], graph: IRGraph) -> bool:
    """Check if all inputs to a node are compile-time known constants.

    Args:
        canonical_inputs (list[str]): List of canonical input IDs.
        graph (IRGraph): Computation graph containing nodes.

    Returns:
        bool: True if every input is an existing Constant node.
    """
    if not canonical_inputs:
        return False
    for inp in canonical_inputs:
        if inp not in graph.nodes or graph.nodes[inp].op_type != "Constant":
            return False
    return True


def _evaluate_constant_node(
    node: IRNode,
    canonical_inputs: list[str],
    graph: IRGraph,
    backend: Any,
) -> Any:
    """Evaluate a constant node using the reference eager backend.

    Args:
        node (IRNode): Primal IR node to fold.
        canonical_inputs (list[str]): Input node IDs.
        graph (IRGraph): Parent computation graph.
        backend (Any): Reference eager backend generator.

    Returns:
        Any: Evaluated concrete value.
    """
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
    outputs = evaluate_graph(subgraph, {}, backend=backend)
    val = outputs[node.id]
    if (hasattr(val, "size") and val.size == 1) or (hasattr(val, "numel") and val.numel() == 1):
        if backend is not None and hasattr(backend, "item"):
            val = backend.item(val)
        elif hasattr(val, "item"):
            val = val.item()
    return val


def constant_folding_pass(graph: IRGraph) -> bool:
    """In-place Constant Folding evaluated via reference eager backend.

    Evaluates compile-time known constant nodes until fixpoint is reached.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: True if any nodes were folded into constants.
    """
    total_modified = False
    pass_modified = True

    while pass_modified:
        pass_modified = False
        sorted_nodes = DAGTopologicalSorter.sort(graph)
        id_map: dict[str, str] = {}
        backend = get_active_backend()
        if backend is None:
            try:
                backend = BackendRegistry.get("numpy")
            except Exception:
                backend = None

        for node in sorted_nodes:
            canonical_inputs = [id_map.get(inp, inp) for inp in node.inputs]
            if node.op_type != "Constant" and _are_all_inputs_constant(canonical_inputs, graph):
                try:
                    val = _evaluate_constant_node(node, canonical_inputs, graph, backend)
                    graph.nodes[node.id].op_type = "Constant"
                    graph.nodes[node.id].attributes = {"value": val}
                    graph.nodes[node.id].inputs = []
                    id_map[node.id] = node.id
                    pass_modified = True
                    total_modified = True
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

    for node in graph.nodes.values():
        for sub in getattr(node, "subgraphs", {}).values():
            if isinstance(sub, LogicalGraph):
                if constant_folding_pass(sub):
                    total_modified = True
        for attr_val in node.attributes.values():
            if isinstance(attr_val, LogicalGraph):
                if constant_folding_pass(attr_val):
                    total_modified = True

    return total_modified
