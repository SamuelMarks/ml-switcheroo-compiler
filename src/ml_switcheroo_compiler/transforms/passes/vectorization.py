# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Vectorization (vmap) middle-end transformation pass."""

from __future__ import annotations

import os
from typing import Any, cast

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node


def _load_vmap_rules() -> dict[str, dict[str, Any]]:
    """Load declarative vectorization rules from YAML configuration.

    Returns:
        dict[str, dict[str, Any]]: Dictionary mapping op types to their vectorization rules.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "vmap_rules.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "rules" in data:
                return cast(dict[str, dict[str, Any]], data["rules"])
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

    # Map tensor ID -> batch axis position (or None if unbatched)
    batch_axis_map: dict[str, int | None] = {}
    shape_map: dict[str, tuple[int, ...]] = {}

    # Initialize batch axes for inputs
    for idx, inp_id in enumerate(graph.inputs):
        axis: int | None = None
        if isinstance(in_axes, dict):
            axis = in_axes.get(inp_id, in_axes.get(str(idx), 0))
        elif isinstance(in_axes, (list, tuple)):
            axis = in_axes[idx] if idx < len(in_axes) else 0
        else:
            axis = in_axes

        batch_axis_map[inp_id] = axis
        orig_shape = ()
        if inp_id in graph.nodes:
            orig_shape = getattr(graph.nodes[inp_id], "shape_metadata", ()) or ()
        if axis is not None:
            new_shape = list(orig_shape)
            if len(new_shape) <= axis:
                new_shape.extend([1] * (axis - len(new_shape) + 1))
            new_shape[axis] = batch_size
            shape_map[inp_id] = tuple(new_shape)
        else:
            shape_map[inp_id] = orig_shape

    for node_id, orig_node in graph.nodes.items():
        if orig_node.op_type == "Input":
            v_node = IRNode(**orig_node.__dict__)
            if node_id in shape_map:
                v_node.shape_metadata = shape_map[node_id]
            vectorized.nodes[node_id] = v_node
            continue

        if orig_node.op_type == "Constant":
            # Constants are initially unbatched
            batch_axis_map[node_id] = None
            shape_map[node_id] = getattr(orig_node, "shape_metadata", ()) or ()
            vectorized.nodes[node_id] = IRNode(**orig_node.__dict__)
            continue

        # Check inputs' batch status
        input_axes = [batch_axis_map.get(inp, None) for inp in orig_node.inputs]
        has_batched_input = any(ax is not None for ax in input_axes)

        if not has_batched_input:
            # Operation remains unbatched
            batch_axis_map[node_id] = None
            shape_map[node_id] = getattr(orig_node, "shape_metadata", ()) or ()
            vectorized.nodes[node_id] = IRNode(**orig_node.__dict__)
            continue

        # At least one input is batched: align batch axes to 0
        aligned_inputs: list[str] = []
        for inp, ax in zip(orig_node.inputs, input_axes):
            cur_shape = shape_map.get(inp, ())
            if ax is not None:
                # Align batch axis to 0 if not already 0
                aligned_id = _align_batch_axis(vectorized, inp, ax, 0, cur_shape)
                aligned_inputs.append(aligned_id)
            else:
                # Unbatched operand: broadcast across batch dimension
                bcast_id = _broadcast_unbatched_input(vectorized, inp, batch_size, cur_shape, 0)
                aligned_inputs.append(bcast_id)

        rule = rules.get(orig_node.op_type, {})
        lift_op = str(rule.get("lift_op", orig_node.op_type))
        policy = str(rule.get("batch_axis_policy", "preserve"))

        new_attrs = dict(orig_node.attributes)
        orig_shape = getattr(orig_node, "shape_metadata", ()) or ()

        if policy == "shift_axis":
            # Shift reduction / softmax axis because of leading batch axis
            if "axis" in new_attrs and isinstance(new_attrs["axis"], int):
                if new_attrs["axis"] >= 0:
                    new_attrs["axis"] += 1
            if "dim" in new_attrs and isinstance(new_attrs["dim"], int):
                if new_attrs["dim"] >= 0:
                    new_attrs["dim"] += 1
            if "axes" in new_attrs and isinstance(new_attrs["axes"], (list, tuple)):
                new_attrs["axes"] = tuple(a + 1 if a >= 0 else a for a in new_attrs["axes"])
            out_shape = (batch_size, *orig_shape) if orig_shape else (batch_size,)
        elif policy == "shift_permutation":
            perm = list(new_attrs.get("permutation", ()))
            new_perm = [0] + [p + 1 for p in perm]
            new_attrs["permutation"] = tuple(new_perm)
            out_shape = (batch_size, *orig_shape) if orig_shape else (batch_size,)
        elif policy == "prepend_batch_dim":
            if "shape" in new_attrs and isinstance(new_attrs["shape"], (list, tuple)):
                new_attrs["shape"] = (batch_size, *new_attrs["shape"])
            if "newshape" in new_attrs and isinstance(new_attrs["newshape"], (list, tuple)):
                new_attrs["newshape"] = (batch_size, *new_attrs["newshape"])
            out_shape = (batch_size, *orig_shape) if orig_shape else (batch_size,)
        elif policy == "batch_matmul":
            lift_op = "BatchMatMul"
            out_shape = (batch_size, *orig_shape) if orig_shape else (batch_size,)
        else:
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

    # Handle outputs and out_axes alignment
    vectorized_outputs: list[str] = []
    target_out_axis = out_axes if isinstance(out_axes, int) else (out_axes[0] if out_axes else 0)

    for out_id in graph.outputs:
        cur_axis = batch_axis_map.get(out_id, 0)
        cur_shape = shape_map.get(out_id, ())
        if cur_axis is not None and cur_axis != target_out_axis:
            aligned_out_id = _align_batch_axis(vectorized, out_id, cur_axis, target_out_axis, cur_shape)
            vectorized_outputs.append(aligned_out_id)
        else:
            vectorized_outputs.append(out_id)

    vectorized.outputs = vectorized_outputs
    return vectorization_pass(vectorized)


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
            body_graph = node.attributes.get("body")
            in_axes = node.attributes.get("in_axes", 0)
            out_axes = node.attributes.get("out_axes", 0)

            # Determine batch size from first batched input
            batch_size = 1
            if node.inputs:
                first_inp = graph.nodes.get(node.inputs[0])
                if first_inp and getattr(first_inp, "shape_metadata", None):
                    axis = in_axes if isinstance(in_axes, int) else in_axes[0]
                    first_shape = first_inp.shape_metadata
                    if axis < len(first_shape):
                        batch_size = first_shape[axis]

            if body_graph and hasattr(body_graph, "nodes"):
                if not isinstance(body_graph, IRGraph):
                    b_graph = IRGraph(name=getattr(body_graph, "id", "body"))
                    b_graph.inputs = list(getattr(body_graph, "inputs", []))
                    b_graph.outputs = list(getattr(body_graph, "outputs", []))
                    b_nodes = body_graph.nodes if isinstance(body_graph.nodes, list) else list(body_graph.nodes.values())
                    for n in b_nodes:
                        b_graph.nodes[n.id] = IRNode(**n.__dict__) if not isinstance(n, IRNode) else n
                    body_graph = b_graph

                v_body = vectorize_graph(body_graph, in_axes=in_axes, batch_size=batch_size, out_axes=out_axes)

                # Connect input mappings
                id_map: dict[str, str] = {}
                for b_in, m_in in zip(v_body.inputs, node.inputs):
                    id_map[b_in] = m_in

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
                continue

        new_nodes[node_id] = node if isinstance(node, IRNode) else IRNode(**node.__dict__)

    if has_vmap:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return graph
