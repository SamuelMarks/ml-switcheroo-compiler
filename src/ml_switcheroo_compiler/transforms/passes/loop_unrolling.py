"""Loop Unrolling pass."""

import os
from typing import Optional, cast

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, LogicalNode, clone_logical_node


def _load_config() -> dict[str, object]:
    """Load the loop unrolling configuration.

    Returns:
        dict[str, object]: The configuration dictionary.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "loop_unrolling.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            res = yaml.safe_load(f)
            return cast(dict[str, object], res) if res else {}
    return {}


def clone_subgraph(graph: IRGraph, prefix: str, id_map: dict[str, str]) -> list[IRNode]:
    """Clone a subgraph for unrolling.

    Args:
        graph (IRGraph): The subgraph to clone.
        prefix (str): Prefix for new node IDs.
        id_map (dict[str, str]): Mapping from old IDs to new IDs.

    Returns:
        list[IRNode]: The cloned nodes.
    """
    cloned_nodes: list[IRNode] = []

    nodes_to_clone = graph.nodes.values() if hasattr(graph, "nodes") else []

    for n in nodes_to_clone:
        new_id = f"{prefix}_{n.id}"
        id_map[n.id] = new_id

        new_inputs = []
        for inp in n.inputs:
            new_inputs.append(id_map.get(inp, inp))

        cloned_node = clone_logical_node(n, id=new_id, inputs=new_inputs)

        if isinstance(n, IRNode):
            # Ensure we convert to IRNode properly, including stream and device
            ir_node = IRNode(**cloned_node.__dict__)
            ir_node.stream = n.stream
            ir_node.device = n.device
            cloned_nodes.append(ir_node)
        else:
            ir_node = IRNode(**cloned_node.__dict__)
            cloned_nodes.append(ir_node)

    return cloned_nodes


def detect_static_bound(node: IRNode, heuristics: list[dict[str, object]]) -> Optional[int]:
    """Detect if a loop has a static bound based on heuristics.

    Args:
        node (IRNode): The node to check.
        heuristics (list[dict[str, object]]): The heuristics to apply.

    Returns:
        Optional[int]: The static bound if found, otherwise None.
    """
    if getattr(node, "op_type", "") != "WhileLoop":
        return None

    attrs = getattr(node, "attributes", {})
    max_iter = attrs.get("max_iterations")
    if isinstance(max_iter, int):
        return max_iter

    for h in heuristics:
        if h.get("op_type") == "WhileLoop" and "max_iterations" in h:
            val = h["max_iterations"]
            if isinstance(val, int):
                return val

    return None


def unroll_loops(graph: IRGraph) -> IRGraph:
    """Unroll WhileLoop IR nodes bounded by statically analyzable iteration counts.

    Args:
        graph (IRGraph): The input graph.

    Returns:
        IRGraph: Graph with unrolled loops.
    """
    config = _load_config()
    default_limit = cast(int, config.get("default_unroll_limit", 10))
    heuristics = cast(list[dict[str, object]], config.get("heuristics", []))

    unrolled = False
    new_nodes: dict[str, LogicalNode] = {}

    for node_id, node in graph.nodes.items():
        if not isinstance(node, IRNode):
            new_nodes[node_id] = node
            continue

        bound = detect_static_bound(node, heuristics)
        if bound is not None and bound <= default_limit:
            unrolled = True

            body_graph = node.attributes.get("body")

            if body_graph and isinstance(body_graph, IRGraph):
                current_state_ids = list(node.inputs)

                for i in range(bound):
                    id_map: dict[str, str] = {}
                    body_inputs = getattr(body_graph, "inputs", [])
                    for b_in, c_in in zip(body_inputs, current_state_ids):
                        id_map[b_in] = c_in

                    cloned = clone_subgraph(body_graph, prefix=f"{node_id}_iter{i}", id_map=id_map)

                    for cn in cloned:
                        new_nodes[cn.id] = cn

                    body_outputs = getattr(body_graph, "outputs", [])
                    current_state_ids = [id_map.get(out, out) for out in body_outputs]

                # Replace loop node with identity pointing to last state
                new_nodes[node_id] = clone_logical_node(node, id=node_id, op_type="Identity", inputs=current_state_ids[:1] if current_state_ids else [])
                new_nodes[node_id] = IRNode(**new_nodes[node_id].__dict__)

                continue

        new_nodes[node_id] = node

    if unrolled:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return graph


def loop_unrolling_pass(graph: IRGraph) -> IRGraph:
    """Pass to unroll static loops in the IR graph.

    Args:
        graph (IRGraph): The input IR graph.

    Returns:
        IRGraph: The optimized IR graph with loops unrolled.
    """
    return unroll_loops(graph)


def _get_initial_constants(*args: object, **kwargs: object) -> list[object]:
    """Get initial constants stub.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        list[object]: Empty list.
    """
    return []
