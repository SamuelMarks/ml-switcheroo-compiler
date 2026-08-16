# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module rewriter.py."""

from typing import Any

"""Shape-Aware Rewriting Passes."""

from ml_switcheroo_ir import LogicalGraph, topological_sort

from ml_switcheroo_compiler.ir.core import clone_logical_node


def shape_aware_rewrite(graph: LogicalGraph) -> LogicalGraph:
    """Resolve structural ambiguities and enforce strict typing using inferred shapes.

    1. Convert dynamic Reshape/View operations (e.g., PyTorch view(-1, dims))
       into explicit explicit shape sequences based on trace shape metadata.
    2. Infer explicit type casts for frameworks requiring strict types (e.g., JAX).

    Args:
        graph (LogicalGraph): The intermediate representation graph.

    Returns:
        LogicalGraph: A new graph with ambiguities resolved.
    """
    new_graph = LogicalGraph(
        name=f"{graph.name}_rewritten",
        outputs=list(graph.outputs),
        mesh=graph.mesh,
    )

    for node in topological_sort(graph):
        new_node = clone_logical_node(node)

        # Rewrite 1: Resolve -1 in Reshape operations
        if node.op_type == "Reshape" and node.shape_metadata is not None:
            new_node.attributes["explicit_shape"] = list(node.shape_metadata)

        # Rewrite 2: Type casts (Mocked for strictly typed backends)
        if node.op_type in ["Add", "Sub", "Mul", "Div", "MatMul"]:
            new_node.attributes["requires_strict_cast"] = True

        new_graph.nodes[node.id] = new_node

    return new_graph
