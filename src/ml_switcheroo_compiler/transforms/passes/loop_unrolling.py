"""Module loop_unrolling.py."""

from __future__ import annotations

"""Loop Unrolling pass for edge execution."""

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

import typing

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


def detect_static_bound(cond_graph: IRBlock, body_graph: IRBlock, initial_state, max_iters: int = 100) -> int | None:
    """Execute detect static bound using lightweight symbolic execution rather than host evaluator.

    Args:
        cond_graph (IRBlock): The cond_graph parameter.
        body_graph (IRBlock): The body_graph parameter.
        initial_state (dict): The initial_state parameter.
        max_iters (int): The max_iters parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    # If the state involves dynamic tensors or symbolic variables, host eager evaluation will break.
    # We must implement pass-through for purely data-dependent conditions and use lightweight static solver.

    # Implement a very lightweight symbolic solver just for constant conditions.
    state = dict(initial_state)

    cond_mock = _block_to_graph(cond_graph)
    body_mock = _block_to_graph(body_graph)

    from ml_switcheroo_compiler.ir.shape_system import SymInt

    def symbolic_eval(g, local_state):
        """symbolic_eval function.

        Args:
        g (object): The g parameter.
        local_state (object): The local_state parameter.

        Returns:
        object: Result.
        """
        from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter

        nodes = DAGTopologicalSorter.sort(g)
        for n in nodes:
            if n.op_type == "Input":
                continue
            elif n.op_type == "Constant":
                local_state[n.id] = n.attributes["value"]
            elif n.op_type == "Add":
                v1 = local_state.get(n.inputs[0], 0)
                v2 = local_state.get(n.inputs[1], 0)
                if isinstance(v1, (int, float, SymInt)) and isinstance(v2, (int, float, SymInt)):
                    local_state[n.id] = v1 + v2
                else:
                    return None
            elif n.op_type == "Sub":
                v1 = local_state.get(n.inputs[0], 0)
                v2 = local_state.get(n.inputs[1], 0)
                if isinstance(v1, (int, float, SymInt)) and isinstance(v2, (int, float, SymInt)):
                    local_state[n.id] = v1 - v2
                else:
                    return None
            elif n.op_type == "Less":
                v1 = local_state.get(n.inputs[0], 0)
                v2 = local_state.get(n.inputs[1], 0)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    local_state[n.id] = v1 < v2
                else:
                    return None
            elif n.op_type == "Output":
                if len(n.inputs) > 1:
                    return tuple(local_state.get(inp, None) for inp in n.inputs)
                return local_state.get(n.inputs[0], None)
            else:
                # Opaque or complex symbolic op, abort unrolling
                return None
        return None

    for i in range(max_iters):
        cond_state = {}
        for c_inp, outer_inp in zip(cond_graph.inputs, initial_state.keys()):  # approximate
            cond_state[c_inp] = state.get(c_inp, state.get(outer_inp))

        res = symbolic_eval(cond_mock, cond_state)
        if res is None:
            return None  # Graceful pass-through
        if not bool(res):
            return i

        body_state = {}
        for b_inp, outer_inp in zip(body_graph.inputs, initial_state.keys()):
            body_state[b_inp] = state.get(b_inp, state.get(outer_inp))

        out_val = symbolic_eval(body_mock, body_state)
        if out_val is None:
            return None

        next_state = {}
        if len(cond_graph.inputs) == 1:
            next_state[cond_graph.inputs[0]] = out_val
        else:
            for j, inp_id in enumerate(cond_graph.inputs):
                next_state[inp_id] = out_val[j] if isinstance(out_val, (list, tuple)) else out_val
        state = next_state

    return None


def _get_initial_constants(node: IRNode, graph: IRGraph):
    """Evaluate _get_initial_constants operation.

    Args:
        node (IRNode): The node parameter.
        graph (IRGraph): The graph parameter.

    Returns:
        dict: Result.
    """
    state = {}
    cond_graph = node.attributes.get("cond")
    if not cond_graph:
        return state

    for outer_inp, inner_inp in zip(node.inputs, cond_graph.inputs):
        outer_node = graph.nodes.get(outer_inp)
        if outer_node and outer_node.op_type == "Constant":
            state[inner_inp] = outer_node.attributes["value"]
    return state


def _perform_unroll(node: IRNode, body_graph: IRGraph, unroll_iters: int, new_nodes) -> None:
    """Perform actual unrolling of a loop body.

    Args:
        node (IRNode): The loop node.
        body_graph (object): The body graph to unroll.
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


def _process_unroll_node(node: IRNode, graph: IRGraph, new_nodes) -> bool:
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
