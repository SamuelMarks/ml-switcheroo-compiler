"""Axis Translation pass for layout conversions."""

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def axis_translation_pass(graph: IRGraph) -> bool:
    """In-place Axis Translation pass.

    Converts operations between layout formats like NCHW and NHWC.

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False

    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        if node.op_type == "Conv2D":
            # Just an example implementation that sets an attribute
            if node.attributes.get("layout") == "NCHW":
                node.attributes["layout"] = "NHWC"
                modified = True

    return modified
