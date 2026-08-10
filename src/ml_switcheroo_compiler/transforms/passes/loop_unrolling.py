from __future__ import annotations

"""Loop Unrolling pass for edge execution."""

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

import typing
from typing import Any

from ml_switcheroo_compiler.ir.core import IRBlock, IRGraph, IRNode, clone_logical_node
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _block_to_graph(block: IRBlock) -> IRGraph:
    """Help to convert IRBlock to IRGraph for standard APIs.

    Args:
        block (IRBlock): The block parameter.

    Returns:
        IRGraph: Result.
    """
    nodes_dict = {n.id: n for n in block.nodes}
    g = IRGraph(nodes=nodes_dict, outputs=block.outputs)
    return g


def clone_subgraph(subgraph: IRBlock, id_suffix: str, input_remap: dict[str, str]) -> tuple[dict[str, IRNode], list[str]]:
    """Clone duplicate a subgraph for unrolling.

    Args:
        subgraph (IRBlock): The input graph to mutate.
        id_suffix (str): Suffix for IDs.
        input_remap (dict): Map of input IDs to outer variables.

    Returns:
        tuple: (cloned_nodes_dict, list_of_output_ids_in_order)
    """
    cloned_nodes = {}
    internal_remap = dict(input_remap)

    mock_graph = _block_to_graph(subgraph)
    sorted_nodes = DAGTopologicalSorter.sort(mock_graph)

    for node in sorted_nodes:
        if node.op_type in ("Input", "Output"):
            continue

        new_id = f"{node.id}_{id_suffix}"
        internal_remap[node.id] = new_id

        new_inputs = [internal_remap.get(inp, inp) for inp in node.inputs]

        cloned = clone_logical_node(node, id=new_id, inputs=new_inputs)
        cloned_nodes[new_id] = cloned

    out_ids = []
    for node in subgraph.nodes:
        if node.op_type == "Output":
            for inp in node.inputs:
                out_ids.append(internal_remap.get(inp, inp))

    return cloned_nodes, out_ids


def detect_static_bound(cond_graph: IRBlock, body_graph: IRBlock, initial_state: dict[str, typing.Any], max_iters: int = 100) -> int | None:
    """Execute detect static bound.

    Args:
        cond_graph (IRBlock): The cond_graph parameter.
        body_graph (IRBlock): The body_graph parameter.
        initial_state (dict): The initial_state parameter.
        max_iters (int): The max_iters parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    state = dict(initial_state)
    backend = get_active_backend()

    cond_mock = _block_to_graph(cond_graph)
    body_mock = _block_to_graph(body_graph)

    for i in range(max_iters):
        try:
            cond_outputs = evaluate_graph(cond_mock, state)
            out_node_id = cond_mock.outputs[0]
            cond_val = cond_outputs[out_node_id]
            is_true = bool(backend.item(cond_val)) if hasattr(cond_val, "size") else bool(cond_val)
        except Exception:
            # print("Cond exc", e)
            return None

        if not is_true:
            return i

        try:
            # Map state to body graph inputs
            body_state: dict[str, typing.Any] = {}
            for c_inp, b_inp in zip(cond_graph.inputs, body_graph.inputs):
                body_state[b_inp] = state.get(c_inp, state.get(b_inp))

            body_outputs = evaluate_graph(body_mock, body_state)
            out_node_id = body_mock.outputs[0]
            out_val = body_outputs[out_node_id]

            next_state: dict[str, typing.Any] = {}
            if len(cond_graph.inputs) == 1:
                next_state[cond_graph.inputs[0]] = out_val
            else:
                for j, inp_id in enumerate(cond_graph.inputs):
                    next_state[inp_id] = typing.cast(list[typing.Any], out_val)[j]
            state = next_state
        except Exception:
            # print("Body exc", e)
            return None

    return None


def _get_initial_constants(node: IRNode, graph: IRGraph) -> dict[str, typing.Any]:
    """Evaluate _get_initial_constants operation.

    Args:
        node (IRNode): The node parameter.
        graph (IRGraph): The graph parameter.

    Returns:
        dict: Result.
    """
    state: dict[str, typing.Any] = {}
    cond_graph = node.attributes.get("cond")
    if not cond_graph:
        return state

    for outer_inp, inner_inp in zip(node.inputs, cond_graph.inputs):
        outer_node = graph.nodes.get(outer_inp)
        if outer_node and outer_node.op_type == "Constant":
            state[inner_inp] = outer_node.attributes["value"]
    return state


def _perform_unroll(node: IRNode, body_graph: Any, unroll_iters: int, new_nodes: dict) -> None:
    """Perform actual unrolling of a loop body.

    Args:
        node (IRNode): The loop node.
        body_graph (Any): The body graph to unroll.
        unroll_iters (int): Number of iterations.
        new_nodes (dict): Target dictionary for new nodes.
    """
    current_inputs = list(node.inputs)

    for i in range(unroll_iters):
        input_remap = {inner: outer for inner, outer in zip(body_graph.inputs, current_inputs)}
        cloned_nodes, next_inputs = clone_subgraph(body_graph, f"unroll_{node.id}_{i}", input_remap)

        new_nodes.update(cloned_nodes)
        current_inputs = next_inputs

    if len(current_inputs) == 1:
        new_nodes[node.id] = IRNode(id=node.id, op_type="Identity", inputs=[current_inputs[0]], shape_metadata=node.shape_metadata)
    else:
        new_nodes[node.id] = IRNode(id=node.id, op_type="Tuple", inputs=current_inputs, shape_metadata=node.shape_metadata)


def _process_unroll_node(node: IRNode, graph: IRGraph, new_nodes: dict) -> bool:
    """Process a node for potential unrolling.

    Args:
        node (IRNode): The IR node.
        graph (IRGraph): The IR graph.
        new_nodes (dict): Output nodes dictionary.

    Returns:
        bool: True if unrolled, False otherwise.
    """
    if node.op_type not in ("WhileLoop", "Loop") or "unrolled" in node.attributes:
        return False

    cond_graph = node.attributes.get("cond")
    body_graph = node.attributes.get("body")

    if not cond_graph or not body_graph:
        return False

    unroll_iters = node.attributes.get("unroll_iters")
    if unroll_iters is None:
        initial_state = _get_initial_constants(node, graph)
        unroll_iters = detect_static_bound(cond_graph, body_graph, initial_state)

    if unroll_iters is not None and unroll_iters > 0:
        _perform_unroll(node, body_graph, unroll_iters, new_nodes)
        return True
    elif unroll_iters == 0:
        if len(node.inputs) == 1:
            new_nodes[node.id] = IRNode(id=node.id, op_type="Identity", inputs=[node.inputs[0]], shape_metadata=node.shape_metadata)
        else:
            new_nodes[node.id] = IRNode(id=node.id, op_type="Tuple", inputs=node.inputs, shape_metadata=node.shape_metadata)
        return True
    else:
        node.attributes["unrolled"] = True
        new_nodes[node.id] = node
        return True


def loop_unrolling_pass(graph: IRGraph) -> bool:
    """In-place Loop Unrolling pass.

    Unrolls loops for lower-level execution targets to reduce branch overhead.

    Args:
        graph (IRGraph): The input graph to mutate.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    modified = False
    new_nodes: dict[str, IRNode] = {}

    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        if _process_unroll_node(node, graph, new_nodes):
            modified = True
        elif node.id not in new_nodes:
            new_nodes[node.id] = node

    if modified:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return modified
