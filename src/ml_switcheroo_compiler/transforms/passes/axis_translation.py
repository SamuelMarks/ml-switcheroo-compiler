# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module axis_translation.py."""

"""Axis Translation pass for layout conversions."""

import uuid

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def axis_translation_pass(graph: IRGraph) -> bool:
    """In-place Axis Translation pass.

    Converts operations between layout formats like NCHW and NHWC by inserting Transpose nodes.

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False

    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        if node.op_type == "Conv2D":
            # For simplicity, let's say the compiler default is NCHW, but edge backends prefer NHWC.
            if node.attributes.get("layout", "NCHW") == "NCHW":
                # Convert the input (N, C, H, W -> N, H, W, C)
                # This requires inserting a transpose BEFORE the Conv2D on its input tensor
                if node.inputs:
                    inp_id = node.inputs[0]
                    transp_in_id = f"transpose_{uuid.uuid4().hex[:8]}"

                    # Estimate new shape metadata
                    orig_shape = graph.nodes[inp_id].shape_metadata if inp_id in graph.nodes else None
                    if orig_shape and isinstance(orig_shape, (tuple, list)) and len(orig_shape) == 4:
                        new_shape = (orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1])
                        new_meta = new_shape
                    else:
                        new_meta = orig_shape

                    transp_in_node = IRNode(id=transp_in_id, op_type="Transpose", inputs=[inp_id], attributes={"axes": [0, 2, 3, 1]}, shape_metadata=new_meta)
                    graph.nodes[transp_in_id] = transp_in_node
                    node.inputs[0] = transp_in_id

                node.attributes["layout"] = "NHWC"

                # Convert the output back (N, H, W, C -> N, C, H, W)
                transp_out_id = f"transpose_{uuid.uuid4().hex[:8]}"

                # The Conv2D now produces NHWC, so we need a Transpose to get it back to NCHW
                transp_out_node = IRNode(id=transp_out_id, op_type="Transpose", inputs=[node.id], attributes={"axes": [0, 3, 1, 2]}, shape_metadata=node.shape_metadata)

                # Update consumers of Conv2D to read from the new Transpose
                for other_node in graph.nodes.values():
                    if other_node.id == transp_out_id:
                        continue
                    if node.id in other_node.inputs:
                        other_node.inputs = [transp_out_id if i == node.id else i for i in other_node.inputs]

                if node.id in graph.outputs:
                    graph.outputs = [transp_out_id if o == node.id else o for o in graph.outputs]

                graph.nodes[transp_out_id] = transp_out_node
                modified = True

    return modified
