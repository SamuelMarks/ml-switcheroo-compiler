"""Shape Inference Pass."""

from __future__ import annotations

import logging
import typing

from ml_switcheroo_compiler.backends.registry import as_int, as_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.transforms.passes.config_models import ShapeInspectionPayload

from collections.abc import Callable

from ml_switcheroo_ir import AttributeValue

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


def _infer_constant_shape(
    node: IRNode,
    shapes: dict[str, tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    """Execute infer constant shape for compiler processing.

    Args:
        node (IRNode): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
        tuple: Result.
    """
    del shapes
    val = node.attributes.get("value")

    backend = get_active_backend()
    arr = backend.array(val)
    return getattr(arr, "shape", ())


def _infer_output_shape(
    node: IRNode,
    shapes: dict[str, tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    """Execute infer output shape for compiler processing.

    Args:
        node (IRNode): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if node.inputs:
        return shapes.get(node.inputs[0])
    return None


def _prepare_op_kwargs(node: IRNode) -> dict[str, AttributeValue]:
    """Execute prepare op kwargs for compiler processing.

    Args:
        node (IRNode): The node parameter.

    Returns:
        dict: Result.
    """
    kwargs: dict[str, AttributeValue] = {**node.attributes}
    if hasattr(node, "shape_metadata") and node.shape_metadata:
        if node.op_type in ("Expand", "BroadcastTo"):
            kwargs["shape"] = node.shape_metadata
        elif node.op_type == "Reshape":
            kwargs["newshape"] = node.shape_metadata
    return kwargs


def _infer_op_shape(
    node: IRNode,
    shapes: dict[str, tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    """Execute infer op shape for compiler processing.

    Args:
        node (IRNode): The node parameter.
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


def _normalize_shape_tuple(raw_shape: tuple[AttributeValue, ...] | list[AttributeValue] | dict[str, AttributeValue] | None) -> tuple[int, ...] | None:
    """Normalize a raw shape container to an optional tuple of ints.

    Args:
        raw_shape (tuple | list | dict | None): Raw shape object.

    Returns:
        tuple[int, ...] | None: Normalized int shape tuple.
    """
    if raw_shape is None:
        return None
    if isinstance(raw_shape, (list, tuple)):
        if any(isinstance(x, str) for x in raw_shape):
            return None
        try:
            return as_shape(tuple(x if hasattr(x, "name") else int(as_int(x)) for x in raw_shape))
        except (ValueError, TypeError):
            return None
    return None


def _infer_input_shape(
    node: IRNode,
    shapes: dict[str, tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    """Infer input node shape from metadata.

    Args:
        node (IRNode): Input node.
        shapes (dict[str, tuple[int, ...] | None]): Known shapes.

    Returns:
        tuple[int, ...] | None: Normalized shape.
    """
    del shapes
    return _normalize_shape_tuple(node.shape_metadata)


_SPECIAL_OP_HANDLERS: dict[
    str,
    Callable[[IRNode, dict[str, tuple[int, ...] | None]], tuple[int, ...] | None],
] = {
    "Constant": _infer_constant_shape,
    "Input": _infer_input_shape,
    "Output": _infer_output_shape,
}


def _determine_node_shape(node: IRNode, shapes: dict[str, tuple[int, ...] | None]) -> tuple[int, ...] | None:
    """Execute determine node shape for compiler processing.

    Args:
        node (IRNode): The node parameter.
        shapes (dict): The shapes parameter.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        CompilationError: An exception.
    """
    if node.op_type in _SPECIAL_OP_HANDLERS:
        return _SPECIAL_OP_HANDLERS[node.op_type](node, shapes)

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


def _propagate_subgraph_shapes(node: IRNode, shapes: dict[str, tuple[int, ...] | None]) -> bool:
    """Propagate input shapes into a node's nested subgraphs and run shape inference on them.

    Args:
        node (IRNode): Parent node containing subgraphs.
        shapes (dict[str, tuple[int, ...] | None]): Known shapes of nodes in the parent graph.

    Returns:
        bool: True if any nested subgraph was modified, False otherwise.
    """
    modified = False
    for sub in getattr(node, "subgraphs", {}).values():
        if hasattr(sub, "nodes") and hasattr(sub, "inputs"):
            for inp_id, parent_inp in zip(sub.inputs, node.inputs):
                if shapes.get(parent_inp) is not None and inp_id in sub.nodes:
                    sub.nodes[inp_id].shape_metadata = shapes[parent_inp]
            if shape_inference_pass(sub):
                modified = True
    return modified


def _fallback_subgraph_output_shape(node: IRNode) -> tuple[int, ...] | None:
    """Attempt to extract output shape from a node's nested subgraphs.

    Args:
        node (IRNode): Node whose subgraphs are inspected.

    Returns:
        tuple[int, ...] | None: Normalized output shape from the first subgraph with an output shape, or None.
    """
    for sub in getattr(node, "subgraphs", {}).values():
        if hasattr(sub, "outputs") and sub.outputs and sub.outputs[0] in sub.nodes:
            sub_out_shape = _normalize_shape_tuple(getattr(sub.nodes[sub.outputs[0]], "shape_metadata", None))
            if sub_out_shape is not None:
                return sub_out_shape
    return None


def _apply_inferred_shape(node: IRNode, out_shape: tuple[int, ...]) -> bool:
    """Update a node's shape metadata and contiguous strides.

    Args:
        node (IRNode): Node to update.
        out_shape (tuple[int, ...]): Inferred concrete shape tuple.

    Returns:
        bool: True if the node's shape_metadata was changed, False otherwise.
    """
    from ml_switcheroo_compiler.ops.shape_inference import compute_contiguous_strides

    modified = False
    if node.shape_metadata != out_shape:
        object.__setattr__(node, "shape_metadata", out_shape)
        modified = True
    strides = compute_contiguous_strides(out_shape)
    node.attributes["strides"] = strides
    try:
        object.__setattr__(node, "strides", strides)
    except (AttributeError, TypeError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return modified


def shape_inference_pass(graph: IRGraph) -> bool:
    """In-place shape inference.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    shapes: dict[str, tuple[int, ...] | None] = {}

    for node in sorted_nodes:
        if _propagate_subgraph_shapes(node, shapes):
            modified = True

        out_shape = _determine_node_shape(node, shapes)
        if out_shape is None:
            out_shape = _fallback_subgraph_output_shape(node)

        shapes[node.id] = out_shape
        if out_shape is not None and _apply_inferred_shape(node, out_shape):
            modified = True

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

    if not hasattr(graph, "annotations") or not isinstance(getattr(graph, "annotations", None), dict):
        object.__setattr__(graph, "annotations", {})

    history: list[dict[str, str | tuple[int, ...]]] = getattr(graph, "annotations").setdefault("learned_shapes_history", [])
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
    shape_map: typing.Mapping[str, typing.Sequence[int]] = {}
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

    if not hasattr(graph, "annotations") or not isinstance(getattr(graph, "annotations", None), dict):
        object.__setattr__(graph, "annotations", {})
    getattr(graph, "annotations")["convergence_converged"] = True

    if modified:
        shape_inference_pass(graph)

    return modified
