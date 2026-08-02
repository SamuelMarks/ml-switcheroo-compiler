"""Batch Norm Folding pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def batch_norm_folding_pass(graph: IRGraph) -> bool:
    """In-place Batch Norm Folding pass.

    Folds BatchNorm operations into preceding Conv2D operations.

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False

    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        if node.op_type == "BatchNorm":
            # Very simplistic folding logic
            if node.inputs:
                prev_node = graph.nodes.get(node.inputs[0])
                if prev_node and prev_node.op_type == "Conv2D" and not node.attributes.get("folded"):
                    prev_node.attributes["folded_batch_norm"] = True
                    # In a real pass we would rewire inputs/outputs
                    # Here we just mark it as modified and perhaps we'd DCE it later
                    node.attributes["folded"] = True
                    modified = True

    return modified
