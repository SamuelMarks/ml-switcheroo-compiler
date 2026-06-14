"""Constant Folding pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def constant_folding_pass(graph: IRGraph) -> bool:
    """In-place Constant Folding.

    Evaluates pure operations on constant inputs eagerly using the interpreter

    graph (IRGraph): The input graph

    Returns:
    bool: True if the graph was modified

    Args:
    graph (IRGraph): Argument graph
    """
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.interpreter import evaluate_graph

    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)

    id_map: dict[str, str] = {}
    backend = get_active_backend()

    for node in sorted_nodes:
        canonical_inputs = [id_map.get(inp, inp) for inp in node.inputs]

        # Check if all inputs are constant
        all_const = True
        for inp in canonical_inputs:
            if inp not in graph.nodes or graph.nodes[inp].op_type != "Constant":
                all_const = False
                break

        if all_const and len(canonical_inputs) > 0:
            # Reconstruct subgraph for just this node
            subgraph = LogicalGraph(outputs=[node.id])
            for inp in canonical_inputs:
                # We need to map IRNode to LogicalNode if they differ in type, but they
                # are aliased
                subgraph.nodes[inp] = graph.nodes[inp]
            subgraph.nodes[node.id] = LogicalNode(
                id=node.id,
                op_type=node.op_type,
                attributes=dict(node.attributes),
                inputs=list(canonical_inputs),
                shape_metadata=node.shape_metadata,
            )

            try:
                # Try evaluating the subgraph
                outputs = evaluate_graph(subgraph, {})
                val = outputs[node.id]

                # Convert single-element arrays to scalar to preserve legacy testing
                # format
                if hasattr(val, "size") and val.size == 1:
                    val = backend.item(val)
                elif hasattr(val, "numel") and val.numel() == 1:
                    val = backend.item(val)

                graph.nodes[node.id].op_type = "Constant"
                graph.nodes[node.id].attributes = {"value": val}
                graph.nodes[node.id].inputs = []
                id_map[node.id] = node.id
                modified = True
                continue
            except Exception:
                # If evaluation fails, fall back to keeping the node
                pass

        id_map[node.id] = node.id

    return modified
