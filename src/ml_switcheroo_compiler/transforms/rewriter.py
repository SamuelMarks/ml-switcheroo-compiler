"""Shape-Aware Rewriting Passes."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode, topological_sort


def shape_aware_rewrite(graph: LogicalGraph) -> LogicalGraph:
    """Resolve structural ambiguities and enforces strict typing using inferred shapes.

    1. Converts dynamic Reshape/View operations (e.g., PyTorch view(-1, dims))
       into explicit explicit shape sequences based on trace shape metadata
    2. Infers explicit type casts for frameworks requiring strict types (e.g., JAX)

    graph (LogicalGraph): The traced input graph

    Returns:
    LogicalGraph: A new graph with ambiguities resolved

    Args:
        graph (LogicalGraph): Argument graph
    """
    new_graph = LogicalGraph(
        name=f"{graph.name}_rewritten",
        outputs=list(graph.outputs),
        mesh=graph.mesh,
    )

    for node in topological_sort(graph):
        new_node = LogicalNode(
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

        # Rewrite 1: Resolve -1 in Reshape operations
        if node.op_type == "Reshape" and node.shape_metadata is not None:
            new_node.attributes["explicit_shape"] = list(node.shape_metadata)

        # Rewrite 2: Type casts (Mocked for strictly typed backends)
        if node.op_type in ["Add", "Sub", "Mul", "Div", "MatMul"]:
            new_node.attributes["requires_strict_cast"] = True

        new_graph.nodes[node.id] = new_node

    return new_graph
