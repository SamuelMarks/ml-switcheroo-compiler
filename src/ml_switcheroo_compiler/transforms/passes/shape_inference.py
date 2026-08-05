"""Shape Inference Pass."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _infer_constant_shape(node: object, shapes: dict) -> tuple:
    """Evaluate _infer_constant_shape operation.

    Args:
        node (object): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
        tuple: Result.
    """
    val = node.attributes.get("value")

    backend = get_active_backend()
    arr = backend.array(val)
    return getattr(arr, "shape", ())


def _infer_output_shape(node: object, shapes: dict) -> tuple | None:
    """Evaluate _infer_output_shape operation.

    Args:
        node (object): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
        object: Result.
    """
    if node.inputs:
        return shapes.get(node.inputs[0])
    return None


def _prepare_op_kwargs(node: object) -> dict:
    """Evaluate _prepare_op_kwargs operation.

    Args:
        node (object): The node parameter.

    Returns:
        dict: Result.
    """
    kwargs = {**node.attributes}
    if hasattr(node, "shape_metadata") and node.shape_metadata:
        if node.op_type in ("Expand", "BroadcastTo"):
            kwargs["shape"] = node.shape_metadata
        elif node.op_type == "Reshape":
            kwargs["newshape"] = node.shape_metadata
    return kwargs


def _infer_op_shape(node: object, shapes: dict) -> tuple | None:
    """Evaluate _infer_op_shape operation.

    Args:
        node (object): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
        object: Result.
    """
    op_cls = get_op(node.op_type)
    op = op_cls()
    in_shapes = [shapes.get(inp) for inp in node.inputs]
    kwargs = _prepare_op_kwargs(node)
    return op.infer_shape(*in_shapes, **kwargs)


def _determine_node_shape(node: IRNode, shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...] | None:
    """Evaluate _determine_node_shape operation.

    Args:
        node (IRNode): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
        object: Result.

    Raises:
        CompilationError: An exception.
    """
    handlers = {
        "Constant": lambda: _infer_constant_shape(node, shapes),
        "Input": lambda: node.shape_metadata,
        "Output": lambda: _infer_output_shape(node, shapes),
    }

    if node.op_type in handlers:
        return handlers[node.op_type]()

    try:
        return _infer_op_shape(node, shapes)
    except KeyError:
        return node.shape_metadata
    except (ValueError, TypeError, Exception) as e:
        msg = f"Shape inference failed at node {node.id} ({node.op_type}): {e!s}"
        raise CompilationError(msg) from e


def shape_inference_pass(graph: IRGraph) -> bool:
    """In-place shape inference.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    shapes = {}

    for node in sorted_nodes:
        out_shape = _determine_node_shape(node, shapes)

        shapes[node.id] = out_shape
        if out_shape is not None and node.shape_metadata != out_shape:
            node.shape_metadata = out_shape
            modified = True

    return modified
