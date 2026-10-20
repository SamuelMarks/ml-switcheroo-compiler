# ruff: noqa: E402
"""Vectorization (vmap) middle-end transformation pass."""

from __future__ import annotations

import os
import typing

import yaml
from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.core.shape import TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node


def _load_vmap_rules() -> dict[str, dict[str, AttributeValue]]:
    """Load declarative vectorization rules from YAML configuration.

    Returns:
        dict[str, dict[str, AttributeValue]]: Dictionary mapping op types to their vectorization rules.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "vmap_rules.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "rules" in data:
                return typing.cast(dict[str, dict[str, AttributeValue]], data["rules"])
    return {}


def _align_batch_axis(
    graph: IRGraph,
    node_id: str,
    current_axis: int,
    target_axis: int,
    shape: tuple[int, ...],
) -> str:
    """Insert a Transpose node to align batch dimension from current_axis to target_axis.

    Args:
        graph (IRGraph): The IR graph to insert the node into.
        node_id (str): The ID of the tensor node whose axis needs alignment.
        current_axis (int): The current position of the batch axis.
        target_axis (int): The target position of the batch axis (typically 0).
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        str: The ID of the aligned tensor node.
    """
    if current_axis == target_axis or len(shape) <= 1:
        return node_id

    rank = len(shape)
    perm = list(range(rank))
    perm.remove(current_axis)
    perm.insert(target_axis, current_axis)

    new_shape = [shape[p] for p in perm]
    transpose_id = f"{node_id}_align_axis_{target_axis}"

    transpose_node = IRNode(
        id=transpose_id,
        op_type="Transpose",
        inputs=[node_id],
        attributes={"permutation": tuple(perm)},
        shape_metadata=tuple(new_shape),
    )
    graph.nodes[transpose_id] = transpose_node
    return transpose_id


def _broadcast_unbatched_input(
    graph: IRGraph,
    node_id: str,
    batch_size: int,
    unbatched_shape: tuple[int, ...],
    target_axis: int = 0,
) -> str:
    """Insert an Expand / BroadcastTo node to broadcast an unbatched operand along the batch dimension.

    Args:
        graph (IRGraph): The IR graph to insert into.
        node_id (str): The ID of the unbatched tensor node.
        batch_size (int): The size of the batch dimension.
        unbatched_shape (tuple[int, ...]): Original unbatched shape.
        target_axis (int): The axis position to broadcast into (defaults to 0).

    Returns:
        str: The ID of the broadcasted tensor node.
    """
    batched_shape = list(unbatched_shape)
    batched_shape.insert(target_axis, batch_size)

    broadcast_id = f"{node_id}_broadcast_batch"
    broadcast_node = IRNode(
        id=broadcast_id,
        op_type="BroadcastTo",
        inputs=[node_id],
        attributes={"shape": tuple(batched_shape)},
        shape_metadata=tuple(batched_shape),
    )
    graph.nodes[broadcast_id] = broadcast_node
    return broadcast_id


def _init_input_batch_axes(
    graph: IRGraph,
    in_axes: int | tuple[int | None, ...] | dict[str, int | None],
    batch_size: int,
) -> tuple[dict[str, int | None], dict[str, tuple[int, ...]]]:
    """Initialize batch axis positions and batched shapes for all graph inputs.

    Args:
        graph (IRGraph): The unbatched input IR graph.
        in_axes (int | tuple[int | None, ...] | dict[str, int | None]): Input batch axis specification.
        batch_size (int): Concrete batch dimension size.

    Returns:
        tuple[dict[str, int | None], dict[str, tuple[int, ...]]]: (batch_axis_map, shape_map).
    """
    batch_axis_map: dict[str, int | None] = {}
    shape_map: dict[str, tuple[int, ...]] = {}

    for idx, inp_id in enumerate(graph.inputs):
        if isinstance(in_axes, dict):
            axis = in_axes.get(inp_id, in_axes.get(str(idx), 0))
        elif isinstance(in_axes, (list, tuple)):
            axis = in_axes[idx] if idx < len(in_axes) else 0
        else:
            axis = in_axes

        batch_axis_map[inp_id] = axis
        orig_shape: tuple[int, ...] = (getattr(graph.nodes[inp_id], "shape_metadata", ()) or ()) if inp_id in graph.nodes else ()
        if axis is not None:
            new_shape = list(orig_shape)
            if len(new_shape) <= axis:
                new_shape.extend([1] * (axis - len(new_shape) + 1))
            new_shape[axis] = batch_size
            shape_map[inp_id] = tuple(new_shape)
        else:
            shape_map[inp_id] = orig_shape

    return batch_axis_map, shape_map


def _apply_batch_axis_policy(
    policy: str,
    lift_op: str,
    new_attrs: dict[str, AttributeValue],
    batch_size: int,
) -> str:
    """Update node attributes and lifted operation type according to the batch axis policy.

    Args:
        policy (str): Batch axis policy name ('shift_axis', 'shift_permutation', 'prepend_batch_dim', 'batch_matmul', 'preserve').
        lift_op (str): Initial lifted operation name.
        new_attrs (dict[str, AttributeValue]): Mutable dictionary of node attributes.
        batch_size (int): Concrete size of the leading batch dimension.

    Returns:
        str: Final lifted operation type name.
    """
    if policy == "shift_axis":
        if isinstance(new_attrs.get("axis"), int) and int(typing.cast(int, new_attrs["axis"])) >= 0:
            new_attrs["axis"] = int(typing.cast(int, new_attrs["axis"])) + 1
        if isinstance(new_attrs.get("dim"), int) and int(typing.cast(int, new_attrs["dim"])) >= 0:
            new_attrs["dim"] = int(typing.cast(int, new_attrs["dim"])) + 1
        if isinstance(new_attrs.get("axes"), (list, tuple)):
            new_attrs["axes"] = tuple(a + 1 if isinstance(a, int) and a >= 0 else a for a in getattr(new_attrs["axes"], "__iter__")())
    elif policy == "shift_permutation":
        perm = list(typing.cast(tuple[int, ...], new_attrs.get("permutation", ())))
        new_attrs["permutation"] = tuple([0] + [p + 1 for p in perm])
    elif policy == "prepend_batch_dim":
        if isinstance(new_attrs.get("shape"), (list, tuple)):
            new_attrs["shape"] = (batch_size, *typing.cast(tuple[int, ...], new_attrs["shape"]))
        if isinstance(new_attrs.get("newshape"), (list, tuple)):
            new_attrs["newshape"] = (batch_size, *typing.cast(tuple[int, ...], new_attrs["newshape"]))
    elif policy == "batch_matmul":
        return "BatchMatMul"
    return lift_op


def _vectorize_single_node(
    node_id: str,
    orig_node: IRNode,
    vectorized: IRGraph,
    batch_axis_map: dict[str, int | None],
    shape_map: dict[str, tuple[int, ...]],
    rules: dict[str, dict[str, AttributeValue]],
    batch_size: int,
) -> None:
    """Vectorize a single node into the target batched graph.

    Args:
        node_id (str): Identifier of the node to vectorize.
        orig_node (IRNode): Original unbatched node.
        vectorized (IRGraph): Target vectorized graph being constructed.
        batch_axis_map (dict[str, int | None]): Mapping of node IDs to batch axis indices.
        shape_map (dict[str, tuple[int, ...]]): Mapping of node IDs to tensor shapes.
        rules (dict[str, dict[str, AttributeValue]]): Loaded vmap rules.
        batch_size (int): Concrete batch dimension size.
    """
    if orig_node.op_type == "Input":
        v_node = IRNode(**orig_node.__dict__)
        if node_id in shape_map:
            v_node.shape_metadata = shape_map[node_id]
        vectorized.nodes[node_id] = v_node
        return

    input_axes = [batch_axis_map.get(inp, None) for inp in orig_node.inputs]
    if orig_node.op_type == "Constant" or not any(ax is not None for ax in input_axes):
        batch_axis_map[node_id] = None
        shape_map[node_id] = getattr(orig_node, "shape_metadata", ()) or ()
        vectorized.nodes[node_id] = IRNode(**orig_node.__dict__)
        return

    aligned_inputs: list[str] = []
    for inp, ax in zip(orig_node.inputs, input_axes):
        cur_shape = shape_map.get(inp, ())
        if ax is not None:
            aligned_inputs.append(_align_batch_axis(vectorized, inp, ax, 0, cur_shape))
        else:
            aligned_inputs.append(_broadcast_unbatched_input(vectorized, inp, batch_size, cur_shape, 0))

    rule = rules.get(orig_node.op_type, {})
    new_attrs = dict(orig_node.attributes)
    lift_op = _apply_batch_axis_policy(
        str(rule.get("batch_axis_policy", "preserve")),
        str(rule.get("lift_op", orig_node.op_type)),
        new_attrs,
        batch_size,
    )
    orig_shape = getattr(orig_node, "shape_metadata", ()) or ()
    out_shape = (batch_size, *orig_shape) if orig_shape else (batch_size,)

    cloned_node = clone_logical_node(
        orig_node,
        id=node_id,
        op_type=lift_op,
        inputs=aligned_inputs,
        attributes=new_attrs,
        shape_metadata=out_shape,
    )
    vectorized.nodes[node_id] = IRNode(**cloned_node.__dict__)
    batch_axis_map[node_id] = 0
    shape_map[node_id] = out_shape


def vectorize_graph(
    graph: IRGraph,
    in_axes: int | tuple[int | None, ...] | dict[str, int | None],
    batch_size: int,
    out_axes: int | tuple[int, ...] = 0,
) -> IRGraph:
    """Rewrite an IRGraph by lifting each operation to its batched equivalent.

    Args:
        graph (IRGraph): The unbatched IR graph to transform.
        in_axes (int | tuple[int | None, ...] | dict[str, int | None]): Batch axis index per input.
        batch_size (int): The concrete batch size.
        out_axes (int | tuple[int, ...]): Expected batch axis index for output(s).

    Returns:
        IRGraph: The transformed, batched IR graph.
    """
    rules = _load_vmap_rules()
    vectorized = IRGraph(name=f"{graph.name}_vectorized")
    vectorized.inputs = list(graph.inputs)

    batch_axis_map, shape_map = _init_input_batch_axes(graph, in_axes, batch_size)

    for node_id, orig_node in graph.nodes.items():
        _vectorize_single_node(node_id, orig_node, vectorized, batch_axis_map, shape_map, rules, batch_size)

    vectorized_outputs: list[str] = []
    target_out_axis = out_axes if isinstance(out_axes, int) else (out_axes[0] if out_axes else 0)

    for out_id in graph.outputs:
        cur_axis = batch_axis_map.get(out_id, 0)
        cur_shape = shape_map.get(out_id, ())
        if cur_axis is not None and cur_axis != target_out_axis:
            vectorized_outputs.append(_align_batch_axis(vectorized, out_id, cur_axis, target_out_axis, cur_shape))
        else:
            vectorized_outputs.append(out_id)

    vectorized.outputs = vectorized_outputs
    return vectorization_pass(vectorized)


def _infer_vmap_batch_size(graph: IRGraph, node: IRNode, in_axes: int | tuple[int, ...]) -> int:
    """Determine the concrete batch size from the first batched input of a Vmap node.

    Args:
        graph (IRGraph): Containing computation graph.
        node (IRNode): The Vmap IRNode.
        in_axes (int | tuple[int, ...]): Batch axis specification for Vmap inputs.

    Returns:
        int: Concrete batch dimension size (defaults to 1).
    """
    first_inp = graph.nodes.get(node.inputs[0]) if node.inputs else None
    first_shape = getattr(first_inp, "shape_metadata", None) if first_inp else None
    if not first_shape:
        return 1
    axis = in_axes if isinstance(in_axes, int) else in_axes[0]
    return int(first_shape[axis]) if axis < len(first_shape) else 1


def _ensure_ir_graph(body_graph: IRGraph | TensorOrScalar) -> IRGraph:
    """Convert a LogicalGraph-like body into an IRGraph if needed.

    Args:
        body_graph (IRGraph | TensorOrScalar): Body graph attribute from a Vmap node.

    Returns:
        IRGraph: Normalized IRGraph instance.
    """
    if isinstance(body_graph, IRGraph):
        return body_graph
    b_graph = IRGraph(name=getattr(body_graph, "id", "body"))
    b_graph.inputs = list(getattr(body_graph, "inputs", []))
    b_graph.outputs = list(getattr(body_graph, "outputs", []))
    raw_nodes = getattr(body_graph, "nodes", [])
    b_nodes = raw_nodes if isinstance(raw_nodes, list) else list(raw_nodes.values())
    for n in b_nodes:
        b_graph.nodes[n.id] = IRNode(**n.__dict__) if type(n) is not IRNode else n
    return b_graph


def _inline_vmap_node(
    graph: IRGraph,
    node_id: str,
    node: IRNode,
    new_nodes: dict[str, IRNode],
) -> bool:
    """Vectorize and inline a Vmap node's body subgraph into new_nodes.

    Args:
        graph (IRGraph): Parent computation graph.
        node_id (str): Identifier of the Vmap node.
        node (IRNode): The Vmap IRNode.
        new_nodes (dict[str, IRNode]): Output dictionary of nodes for the lowered graph.

    Returns:
        bool: True if the Vmap node had a valid body and was inlined, False otherwise.
    """
    body_graph = node.attributes.get("body")
    if not body_graph or not hasattr(body_graph, "nodes"):
        return False

    in_axes = node.attributes.get("in_axes", 0)
    out_axes = node.attributes.get("out_axes", 0)
    batch_size = typing.cast("TypedVariadicCallable[int]", _infer_vmap_batch_size)(graph, node, in_axes)
    v_body = typing.cast("TypedVariadicCallable[IRGraph]", vectorize_graph)(_ensure_ir_graph(typing.cast("LogicalGraph", body_graph)), in_axes=in_axes, batch_size=batch_size, out_axes=out_axes)

    id_map: dict[str, str] = dict(zip(v_body.inputs, node.inputs))
    for b_id, b_node in v_body.nodes.items():
        if b_node.op_type == "Input":
            continue
        new_b_id = f"{node_id}_{b_id}"
        id_map[b_id] = new_b_id
        new_inputs = [id_map.get(inp, inp) for inp in b_node.inputs]
        cloned = clone_logical_node(b_node, id=new_b_id, inputs=new_inputs)
        new_nodes[new_b_id] = IRNode(**cloned.__dict__)

    out_mapped = [id_map.get(o, o) for o in v_body.outputs]
    identity_node = clone_logical_node(node, id=node_id, op_type="Identity", inputs=out_mapped[:1], attributes={})
    new_nodes[node_id] = IRNode(**identity_node.__dict__)
    return True


def vectorization_pass(graph: IRGraph) -> IRGraph:
    """Middle-end pass that detects Vmap nodes in an IRGraph and inlines their vectorized subgraphs.

    Args:
        graph (IRGraph): The input IR graph potentially containing Vmap nodes.

    Returns:
        IRGraph: The optimized graph with Vmap nodes lowered to batched operations.
    """
    new_nodes: dict[str, IRNode] = {}
    has_vmap = False

    for node_id, node in graph.nodes.items():
        if getattr(node, "op_type", "") == "Vmap":
            has_vmap = True
            if _inline_vmap_node(graph, node_id, node, new_nodes):
                continue

        new_nodes[node_id] = node if type(node) is IRNode else IRNode(**node.__dict__)

    if has_vmap:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return graph


from ml_switcheroo_ir import AttributeValue
