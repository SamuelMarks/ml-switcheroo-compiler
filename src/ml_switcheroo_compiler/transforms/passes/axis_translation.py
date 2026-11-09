"""Axis Translation pass for layout conversions."""

from __future__ import annotations

import uuid

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _transpose_conv_weight(graph: IRGraph, node: IRNode) -> None:
    """Insert a `Transpose` node on the 4D convolutional filter input (`OIHW -> HWIO`).

    Args:
        graph (IRGraph): The computation graph to mutate.
        node (IRNode): The `Conv2D` node being translated from `NCHW` to `NHWC`.
    """
    if len(node.inputs) < 2:
        return
    w_id = node.inputs[1]
    transp_w_id = f"transpose_w_{uuid.uuid4().hex[:8]}"
    w_shape = graph.nodes[w_id].shape_metadata if w_id in graph.nodes else None
    new_w_meta = (w_shape[2], w_shape[3], w_shape[1], w_shape[0]) if w_shape and len(w_shape) == 4 else w_shape
    transp_w_node = IRNode(
        id=transp_w_id,
        op_type="Transpose",
        inputs=[w_id],
        attributes={"axes": [2, 3, 1, 0]},
        shape_metadata=new_w_meta,
    )
    graph.nodes[transp_w_id] = transp_w_node
    node.inputs[1] = transp_w_id
    node.attributes["kernel_layout"] = "HWIO"


def axis_translation_pass(graph: IRGraph, target_layout: str = "NHWC") -> bool:
    """In-place Axis Translation pass.

    Converts operations between layout formats like NCHW and NHWC by inserting Transpose nodes.

    Args:
        graph (IRGraph): The input graph to mutate.
        target_layout (str): Desired target spatial layout (defaults to `'NHWC'`).

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False

    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        if node.op_type == "Conv2D":
            current_layout = str(node.attributes.get("layout", "NCHW"))
            if current_layout == "NCHW" and target_layout == "NHWC":
                # Convert the input (N, C, H, W -> N, H, W, C)
                if node.inputs:
                    inp_id = node.inputs[0]
                    transp_in_id = f"transpose_{uuid.uuid4().hex[:8]}"

                    orig_shape = graph.nodes[inp_id].shape_metadata if inp_id in graph.nodes else None
                    new_meta = (orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1]) if orig_shape and len(orig_shape) == 4 else orig_shape

                    transp_in_node = IRNode(id=transp_in_id, op_type="Transpose", inputs=[inp_id], attributes={"axes": [0, 2, 3, 1]}, shape_metadata=new_meta)
                    graph.nodes[transp_in_id] = transp_in_node
                    node.inputs[0] = transp_in_id

                _transpose_conv_weight(graph, node)
                node.attributes["layout"] = target_layout

                # Convert the output back (N, H, W, C -> N, C, H, W)
                transp_out_id = f"transpose_{uuid.uuid4().hex[:8]}"
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
