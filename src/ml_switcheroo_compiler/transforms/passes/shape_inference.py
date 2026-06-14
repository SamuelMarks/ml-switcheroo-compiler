"""Shape Inference Pass."""

from __future__ import annotations

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.ops import get_op
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _infer_constant_shape(node: object, shapes: dict) -> tuple:
    """Execute _infer_constant_shape.

    Args:
        node (Any): Argument node.
        shapes (Any): Argument shapes.

    Returns:
    Any: The result.
    """
    val = node.attributes.get("value")
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    arr = backend.array(val)
    return getattr(arr, "shape", ())


def _infer_output_shape(node: object, shapes: dict) -> tuple | None:
    """Execute _infer_output_shape.

    Args:
        node (Any): Argument node.
        shapes (Any): Argument shapes.

    Returns:
    Any: The result.
    """
    if node.inputs:
        return shapes.get(node.inputs[0])
    return None


def _infer_op_shape(node: object, shapes: dict) -> tuple | None:
    """Execute _infer_op_shape.

    Args:
        node (Any): Argument node.
        shapes (Any): Argument shapes.

    Returns:
    Any: The result.
    """
    op_cls = get_op(node.op_type)
    op = op_cls()
    in_shapes = [shapes.get(inp) for inp in node.inputs]
    kwargs = {**node.attributes}
    if hasattr(node, "shape_metadata") and node.shape_metadata:
        if node.op_type in ("Expand", "BroadcastTo"):
            kwargs["shape"] = node.shape_metadata
        elif node.op_type == "Reshape":
            kwargs["newshape"] = node.shape_metadata
    return op.infer_shape(*in_shapes, **kwargs)


def shape_inference_pass(graph: IRGraph) -> bool:
    """In-place shape inference.

    Updates node.shape_metadata based on operations

    Args:
        graph (IRGraph): Argument graph

    Returns:
    bool: True if modified
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    shapes = {}

    for node in sorted_nodes:
        if node.op_type == "Constant":
            out_shape = _infer_constant_shape(node, shapes)
        elif node.op_type == "Input":
            out_shape = node.shape_metadata
        elif node.op_type == "Output":
            out_shape = _infer_output_shape(node, shapes)
        else:
            try:
                out_shape = _infer_op_shape(node, shapes)
            except KeyError:
                out_shape = node.shape_metadata
            except (ValueError, TypeError, NotImplementedError) as e:
                msg = f"Shape inference failed at node {node.id} ({node.op_type}): {e!s}"
                raise CompilationError(msg) from e

        shapes[node.id] = out_shape
        if out_shape is not None and node.shape_metadata != out_shape:
            node.shape_metadata = out_shape
            modified = True

    return modified
