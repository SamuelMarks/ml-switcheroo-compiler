# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module batch_norm_folding.py."""

from typing import Any

"""Batch Norm Folding pass."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def batch_norm_folding_pass(graph: IRGraph) -> bool:
    """In-place Batch Norm Folding pass.

    Folds BatchNorm operations into preceding Conv2D operations by rewiring the graph.

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False

    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        if node.op_type == "BatchNorm":
            if node.inputs:
                prev_node = graph.nodes.get(node.inputs[0])
                if prev_node and prev_node.op_type == "Conv2D":
                    # Collect batchnorm inputs (moving mean, var, scale, bias)
                    # We conceptually fold them by modifying the Conv2D node to a FusedConv2DBatchNorm
                    # For simplicity, we just rewire consumers of BatchNorm to point directly to Conv2D,
                    # and mark Conv2D with folded attributes (actual math folding happens in emitters/backends)

                    prev_node.attributes["folded_batch_norm"] = True
                    prev_node.attributes["bn_inputs"] = node.inputs[1:]  # scale, bias, mean, var

                    # Rewire consumers of BatchNorm to point to Conv2D
                    for other_node in graph.nodes.values():
                        if node.id in other_node.inputs:
                            other_node.inputs = [prev_node.id if i == node.id else i for i in other_node.inputs]
                            modified = True

                    if node.id in graph.outputs:
                        graph.outputs = [prev_node.id if o == node.id else o for o in graph.outputs]
                        modified = True

                    # Remove the batch norm node from the graph
                    del graph.nodes[node.id]
                    modified = True

    return modified
