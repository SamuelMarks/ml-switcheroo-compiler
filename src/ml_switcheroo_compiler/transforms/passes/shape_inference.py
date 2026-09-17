"""Module shape_inference.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Shape Inference Pass."""


import typing

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.transforms.passes.config_models import ShapeInspectionPayload

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.errors import CompilationError, ShapeMismatchError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter
from ml_switcheroo_compiler.transforms.passes.config_models import (
    RuntimeShapePacket,
    ShapeInspectionPayload,
    ShapeLearningProtocolConfig,
    load_shape_learning_protocol,
)


def _infer_constant_shape(node, shapes):
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


def _infer_output_shape(node, shapes):
    """Evaluate _infer_output_shape operation.

    Args:
        node (object): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if node.inputs:
        return shapes.get(node.inputs[0])
    return None


def _prepare_op_kwargs(node):
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


def _infer_op_shape(node, shapes):
    """Evaluate _infer_op_shape operation.

    Args:
        node (object): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    op_cls = get_op(node.op_type)
    op = op_cls()
    in_shapes = [shapes.get(inp) for inp in node.inputs]
    kwargs = _prepare_op_kwargs(node)
    result = op.infer_shape(*in_shapes, **kwargs)
    return result if isinstance(result, tuple) else None


def _normalize_shape_tuple(raw_shape: object) -> tuple[int, ...] | None:
    """Normalize a raw shape container to an optional tuple of ints.

    Args:
        raw_shape (object): Raw shape object.

    Returns:
        tuple[int, ...] | None: Normalized int shape tuple.
    """
    if raw_shape is None:
        return None
    if isinstance(raw_shape, (list, tuple)):
        from ml_switcheroo_compiler.ops.shape_inference import S

        if any(isinstance(x, str) for x in raw_shape):
            return None
        try:
            return tuple(int(x) if not isinstance(x, S) else x for x in raw_shape)
        except (ValueError, TypeError):
            return None
    return None


def _determine_node_shape(node: IRNode, shapes: dict[str, tuple[int, ...] | None]) -> tuple[int, ...] | None:
    """Evaluate _determine_node_shape operation.

    Args:
        node (IRNode): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        CompilationError: An exception.
    """
    handlers = {
        "Constant": lambda: _infer_constant_shape(node, shapes),
        "Input": lambda: _normalize_shape_tuple(node.shape_metadata),
        "Output": lambda: _infer_output_shape(node, shapes),
    }

    if node.op_type in handlers:
        return handlers[node.op_type]()

    try:
        return _infer_op_shape(node, shapes)
    except (KeyError, AttributeError):
        return _normalize_shape_tuple(node.shape_metadata)
    except ValueError as e:
        if "Operation" in str(e) and "not found" in str(e):
            return _normalize_shape_tuple(node.shape_metadata)
        msg = f"Shape inference failed at node {node.id} ({node.op_type}): {e!s}"
        raise CompilationError(msg) from e
    except (TypeError, Exception) as e:
        msg = f"Shape inference failed at node {node.id} ({node.op_type}): {e!s}"
        raise CompilationError(msg) from e


def shape_inference_pass(graph: IRGraph) -> bool:
    """In-place shape inference.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    from ml_switcheroo_compiler.ops.shape_inference import compute_contiguous_strides

    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    shapes: dict[str, tuple[int, ...] | None] = {}

    for node in sorted_nodes:
        # Propagate shapes across nested subgraphs
        for sub in getattr(node, "subgraphs", {}).values():
            if hasattr(sub, "nodes") and hasattr(sub, "inputs"):
                for inp_id, parent_inp in zip(sub.inputs, node.inputs):
                    if parent_inp in shapes and shapes[parent_inp] is not None:
                        if inp_id in sub.nodes:
                            sub.nodes[inp_id].shape_metadata = shapes[parent_inp]
                if shape_inference_pass(sub):
                    modified = True

        out_shape = _determine_node_shape(node, shapes)

        if out_shape is None:
            for sub in getattr(node, "subgraphs", {}).values():
                if hasattr(sub, "outputs") and sub.outputs and sub.outputs[0] in sub.nodes:
                    sub_out_shape = _normalize_shape_tuple(getattr(sub.nodes[sub.outputs[0]], "shape_metadata", None))
                    if sub_out_shape is not None:
                        out_shape = sub_out_shape
                        break

        shapes[node.id] = out_shape
        if out_shape is not None:
            if node.shape_metadata != out_shape:
                node.shape_metadata = out_shape
                modified = True
            strides = compute_contiguous_strides(out_shape)
            node.attributes["strides"] = strides
            try:
                node.strides = strides
            except AttributeError:
                pass

    return modified


def get_shape_learning_protocol_config() -> ShapeLearningProtocolConfig:
    """Retrieve the declarative shape learning protocol specification.

    Returns:
        ShapeLearningProtocolConfig: Loaded protocol configuration.
    """
    return load_shape_learning_protocol()


def record_runtime_observation(
    graph: IRGraph,
    node_id: str,
    observed_shape: typing.Sequence[int],
    observed_dtype: str = "float32",
) -> None:
    """Bind an empirical runtime shape observation to a specific graph node.

    Validates that observed rank and static constraints match prior knowledge,
    recording observation history in graph annotations.

    Args:
        graph (IRGraph): The target computation graph.
        node_id (str): The node identifier in the graph.
        observed_shape (typing.Sequence[int]): Concrete dimensions captured at runtime.
        observed_dtype (str): Observed tensor data type.

    Raises:
        KeyError: If node_id does not exist in the graph.
        ShapeMismatchError: If the observed shape violates static rank or dimension invariants.
    """
    if node_id not in graph.nodes:
        raise KeyError(f"Node '{node_id}' not found in computation graph.")

    node: IRNode = graph.nodes[node_id]
    norm_shape: tuple[int, ...] = tuple(int(s) for s in observed_shape)

    current_shape = getattr(node, "shape_metadata", None)
    if current_shape is not None and isinstance(current_shape, (tuple, list)):
        if len(current_shape) != len(norm_shape):
            raise ShapeMismatchError(f"Rank mismatch for node '{node_id}': expected rank {len(current_shape)}, but observed rank {len(norm_shape)} (shape: {norm_shape}).")

    node.shape_metadata = norm_shape

    if not hasattr(graph, "annotations") or not isinstance(graph.annotations, dict):
        graph.annotations = {}

    history: list[dict[str, str | tuple[int, ...]]] = graph.annotations.setdefault("learned_shapes_history", [])
    history.append(
        {
            "node_id": node_id,
            "shape": norm_shape,
            "dtype": observed_dtype,
        }
    )


def annotate_learned_shapes(
    graph: IRGraph,
    observed_shapes: (dict[str, tuple[int, ...]] | dict[str, list[int]] | ShapeInspectionPayload | RuntimeShapePacket),
) -> bool:
    """Annotate graph nodes with concrete runtime shapes captured during browser or hardware execution.

    Args:
        graph (IRGraph): The computation graph to annotate with learned shapes.
        observed_shapes (Union[dict[str, tuple[int, ...]], dict[str, list[int]], ShapeInspectionPayload, RuntimeShapePacket]):
            Mapping of node IDs to observed concrete shapes or full runtime payload.

    Returns:
        bool: True if any node shape metadata was updated, False otherwise.
    """
    shape_map: dict[str, typing.Sequence[int]] = {}
    if isinstance(observed_shapes, RuntimeShapePacket):
        shape_map = {obs.node_id: obs.shape for obs in observed_shapes.observations}
    elif isinstance(observed_shapes, ShapeInspectionPayload):
        shape_map = observed_shapes.observed_shapes
    elif isinstance(observed_shapes, dict):
        shape_map = observed_shapes

    modified: bool = False
    for nid, shape in shape_map.items():
        if nid in graph.nodes:
            prev_shape = getattr(graph.nodes[nid], "shape_metadata", None)
            norm_shape = tuple(int(s) for s in shape)
            if prev_shape != norm_shape:
                record_runtime_observation(graph, nid, shape)
                modified = True

    if not hasattr(graph, "annotations") or not isinstance(graph.annotations, dict):
        graph.annotations = {}
    graph.annotations["convergence_converged"] = True

    if modified:
        shape_inference_pass(graph)

    return modified
