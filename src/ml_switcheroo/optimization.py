"""Optimization passes for LogicalGraph."""

from typing import Dict, Set

from ml_switcheroo_ir import LogicalGraph, LogicalNode, topological_sort


def dce(graph: LogicalGraph) -> LogicalGraph:
    """Dead Code Elimination (DCE).

    Removes nodes that do not contribute to the graph outputs.

    Args:
        graph (LogicalGraph): The input graph.

    Returns:
        LogicalGraph: The optimized graph.
    """
    reachable: Set[str] = set(graph.outputs)

    # Simple backward reachability
    sorted_nodes = topological_sort(graph)
    for node in reversed(sorted_nodes):
        if node.id in reachable:
            for inp in node.inputs:
                reachable.add(inp)

    new_graph = LogicalGraph(
        name=f"{graph.name}_dce", outputs=list(graph.outputs), mesh=graph.mesh
    )
    for nid, node in graph.nodes.items():
        if nid in reachable:
            # We don't deepcopy fully, just structural copy to avoid modifying original
            new_graph.nodes[nid] = LogicalNode(
                id=node.id,
                op_type=node.op_type,
                domain=node.domain,
                version=node.version,
                attributes=dict(node.attributes),
                inputs=list(node.inputs),
                shape_metadata=node.shape_metadata,
                source_ast_ref=node.source_ast_ref,
                sharding=node.sharding,
            )

    return new_graph


def cse(graph: LogicalGraph) -> LogicalGraph:
    """Common Subexpression Elimination (CSE).

    Merges nodes that perform the same operation with the same inputs and attributes.

    Args:
        graph (LogicalGraph): The input graph.

    Returns:
        LogicalGraph: The optimized graph.
    """
    new_graph = LogicalGraph(name=f"{graph.name}_cse", mesh=graph.mesh)

    # Hash of node signature -> canonical node id
    seen_expressions: Dict[str, str] = {}
    id_map: Dict[str, str] = {}

    sorted_nodes = topological_sort(graph)

    for node in sorted_nodes:
        # Map inputs to canonical inputs
        canonical_inputs = tuple(id_map.get(inp, inp) for inp in node.inputs)

        # Serialize attributes deterministically
        attr_str = str(sorted([(k, str(v)) for k, v in node.attributes.items()]))

        signature = f"{node.op_type}|{canonical_inputs}|{attr_str}"

        if signature in seen_expressions:
            id_map[node.id] = seen_expressions[signature]
        else:
            seen_expressions[signature] = node.id
            id_map[node.id] = node.id

            new_graph.nodes[node.id] = LogicalNode(
                id=node.id,
                op_type=node.op_type,
                domain=node.domain,
                version=node.version,
                attributes=dict(node.attributes),
                inputs=list(canonical_inputs),
                shape_metadata=node.shape_metadata,
                source_ast_ref=node.source_ast_ref,
                sharding=node.sharding,
            )

    # Update outputs to point to canonical nodes
    new_graph.outputs = [id_map.get(o, o) for o in graph.outputs]
    return new_graph


def constant_folding(graph: LogicalGraph) -> LogicalGraph:
    """Constant Folding.

    Evaluates pure operations on constant inputs eagerly.
    (Currently a simplified placeholder for pure-Python fallback evaluation).

    Args:
        graph (LogicalGraph): The input graph.

    Returns:
        LogicalGraph: The optimized graph.
    """
    new_graph = LogicalGraph(name=f"{graph.name}_fold", mesh=graph.mesh)
    sorted_nodes = topological_sort(graph)

    id_map: Dict[str, str] = {}

    for node in sorted_nodes:
        canonical_inputs = [id_map.get(inp, inp) for inp in node.inputs]

        # Check if all inputs are constant
        all_const = True
        input_values = []
        for inp in canonical_inputs:
            if inp not in new_graph.nodes or new_graph.nodes[inp].op_type != "Constant":
                all_const = False
                break
            input_values.append(new_graph.nodes[inp].attributes.get("value"))

        if (
            all_const
            and len(canonical_inputs) > 0
            and node.op_type in ["Add", "Sub", "Mul", "Div"]
        ):
            # Evaluate using standard python ops for scalars as placeholder
            try:
                if node.op_type == "Add":
                    val = input_values[0] + input_values[1]
                elif node.op_type == "Sub":
                    val = input_values[0] - input_values[1]
                elif node.op_type == "Mul":
                    val = input_values[0] * input_values[1]
                else:  # node.op_type == "Div"
                    val = input_values[0] / input_values[1]

                # Create a new constant node
                new_graph.nodes[node.id] = LogicalNode(
                    id=node.id,
                    op_type="Constant",
                    attributes={"value": val},
                    shape_metadata=node.shape_metadata,
                )
                id_map[node.id] = node.id
                continue
            except Exception:
                # If evaluation fails, fall back to keeping the node
                pass

        # Keep node as-is if not folded
        new_graph.nodes[node.id] = LogicalNode(
            id=node.id,
            op_type=node.op_type,
            domain=node.domain,
            version=node.version,
            attributes=dict(node.attributes),
            inputs=list(canonical_inputs),
            shape_metadata=node.shape_metadata,
            source_ast_ref=node.source_ast_ref,
            sharding=node.sharding,
        )
        id_map[node.id] = node.id

    new_graph.outputs = [id_map.get(o, o) for o in graph.outputs]
    return new_graph
