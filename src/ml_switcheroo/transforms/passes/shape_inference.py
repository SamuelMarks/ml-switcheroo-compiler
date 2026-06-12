"""Shape Inference Pass."""

from ml_switcheroo.ir.core import IRGraph
from ml_switcheroo.transforms.pass_manager import DAGTopologicalSorter
from ml_switcheroo.core.errors import CompilationError
from ml_switcheroo.ops import get_op


def shape_inference_pass(graph: IRGraph) -> bool:
    """In-place shape inference.

    Updates node.shape_metadata based on operations.

    Args:
        graph: The IR graph.

    Returns:
        bool: True if modified.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)

    # Store shapes dictionary
    shapes = {}

    for node in sorted_nodes:
        if node.op_type == "Constant":
            val = node.attributes.get("value")
            import numpy as np

            shape = np.array(val).shape
            shapes[node.id] = shape
            if node.shape_metadata != shape:
                modified = True
            node.shape_metadata = shape
            continue

        if node.op_type == "Input":
            shapes[node.id] = node.shape_metadata
            continue

        if node.op_type == "Output":
            # For output, inherit shape from its input if available
            inp_shape = None
            if node.inputs:
                inp_shape = shapes.get(node.inputs[0])
            shapes[node.id] = inp_shape
            if node.shape_metadata != inp_shape:
                modified = True
            node.shape_metadata = inp_shape
            continue

        try:
            op_cls = get_op(node.op_type)
            op = op_cls()

            in_shapes = [shapes.get(inp) for inp in node.inputs]

            # Extract kwargs for shape inference
            kwargs = {**node.attributes}
            if hasattr(node, "shape_metadata") and node.shape_metadata:
                # Some ops like Expand/BroadcastTo might store the target shape in
                # shape_metadata
                if node.op_type in ("Expand", "BroadcastTo"):
                    kwargs["shape"] = node.shape_metadata
                elif node.op_type == "Reshape":
                    kwargs["newshape"] = node.shape_metadata

            out_shape = op.infer_shape(*in_shapes, **kwargs)
            shapes[node.id] = out_shape

            if out_shape is not None and node.shape_metadata != out_shape:
                node.shape_metadata = out_shape
                modified = True

        except KeyError:
            # Unknown ops default to None or preserving existing metadata
            shapes[node.id] = node.shape_metadata

        except Exception as e:
            raise CompilationError(
                f"Shape inference failed at node {node.id} ({node.op_type}): {str(e)}"
            ) from e

    return modified
