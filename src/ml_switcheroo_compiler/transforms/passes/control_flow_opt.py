# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Control flow optimizations: branch pruning, invariant hoisting, and symbolic predicate evaluation."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node


def evaluate_static_predicate(graph: IRGraph, pred_id: str) -> bool | None:
    """Evaluate whether a predicate node represents a statically known boolean constant.

    Args:
        graph (IRGraph): The containing IR graph.
        pred_id (str): The ID of the predicate node.

    Returns:
        Optional[bool]: Boolean value if statically evaluable, else None.
    """
    node = graph.nodes.get(pred_id)
    if node is None:
        return None

    if getattr(node, "op_type", "") == "Constant":
        val = node.attributes.get("value")
        if isinstance(val, (bool, int, float)):
            return bool(val)
        if hasattr(val, "item"):
            return bool(val.item())

    # Check comparison of constants
    cmp_dispatch = {
        "Equal": lambda a, b: a == b,
        "Less": lambda a, b: a < b,
        "Greater": lambda a, b: a > b,
        "LessEqual": lambda a, b: a <= b,
        "GreaterEqual": lambda a, b: a >= b,
    }
    if getattr(node, "op_type", "") in cmp_dispatch:
        if len(node.inputs) >= 2:
            in0 = graph.nodes.get(node.inputs[0])
            in1 = graph.nodes.get(node.inputs[1])
            if in0 and in1 and in0.op_type == "Constant" and in1.op_type == "Constant":
                v0 = in0.attributes.get("value")
                v1 = in1.attributes.get("value")
                if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                    return bool(cmp_dispatch[node.op_type](v0, v1))

    return None


def _hoist_common_invariants(
    outer_graph: IRGraph,
    cond_node_id: str,
    true_branch: Any,
    false_branch: Any,
) -> list[IRNode]:
    """Identify and hoist identical computations from both branches to the outer graph.

    Args:
        outer_graph (IRGraph): The outer enclosing graph.
        cond_node_id (str): The ID of the conditional node.
        true_branch (Any): The true / then branch graph or block.
        false_branch (Any): The false / else branch graph or block.

    Returns:
        list[IRNode]: Hoisted nodes added to the outer graph.
    """
    hoisted_nodes: list[IRNode] = []
    t_nodes = true_branch.nodes if isinstance(true_branch.nodes, dict) else {n.id: n for n in getattr(true_branch, "nodes", [])}
    f_nodes = false_branch.nodes if isinstance(false_branch.nodes, dict) else {n.id: n for n in getattr(false_branch, "nodes", [])}

    # Find common operations on external outer inputs
    for t_id, t_node in list(t_nodes.items()):
        if getattr(t_node, "op_type", "") in ("Input", "Output"):
            continue
        for f_id, f_node in list(f_nodes.items()):
            if getattr(f_node, "op_type", "") in ("Input", "Output"):
                continue
            if t_node.op_type == f_node.op_type and t_node.inputs == f_node.inputs:
                # All inputs must be present in the outer graph
                if all(inp in outer_graph.nodes for inp in t_node.inputs):
                    hoist_id = f"{cond_node_id}_hoisted_{t_node.op_type.lower()}"
                    hoisted = IRNode(
                        id=hoist_id,
                        op_type=t_node.op_type,
                        inputs=list(t_node.inputs),
                        attributes=dict(t_node.attributes),
                        shape_metadata=getattr(t_node, "shape_metadata", ()),
                    )
                    hoisted_nodes.append(hoisted)

                    # Update branch nodes to be identities pointing to hoisted node
                    t_nodes[t_id] = IRNode(
                        id=t_id,
                        op_type="Identity",
                        inputs=[hoist_id],
                        attributes={},
                        shape_metadata=getattr(t_node, "shape_metadata", ()),
                    )
                    f_nodes[f_id] = IRNode(
                        id=f_id,
                        op_type="Identity",
                        inputs=[hoist_id],
                        attributes={},
                        shape_metadata=getattr(f_node, "shape_metadata", ()),
                    )
                    break

    return hoisted_nodes


def control_flow_optimization_pass(graph: IRGraph) -> IRGraph:
    """Run control flow optimizations on Cond/If nodes: branch pruning and invariant hoisting.

    Args:
        graph (IRGraph): The input IR graph.

    Returns:
        IRGraph: The optimized IR graph.
    """
    new_nodes: dict[str, IRNode] = {}
    modified = False

    for node_id, node in graph.nodes.items():
        op_type = getattr(node, "op_type", "")
        if op_type in ("Cond", "If"):
            pred_id = node.inputs[0] if node.inputs else None
            true_branch = node.attributes.get("then_branch", node.attributes.get("true_branch"))
            false_branch = node.attributes.get("else_branch", node.attributes.get("false_branch"))

            # Step 1: Symbolic predicate evaluation & branch pruning
            static_pred = evaluate_static_predicate(graph, pred_id) if pred_id else None
            if static_pred is not None and (true_branch or false_branch):
                modified = True
                active_branch = true_branch if static_pred else false_branch

                if active_branch:
                    b_nodes = active_branch.nodes if isinstance(active_branch.nodes, dict) else {n.id: n for n in getattr(active_branch, "nodes", [])}
                    id_map: dict[str, str] = {}

                    # Inline active branch
                    for b_id, b_node in b_nodes.items():
                        if getattr(b_node, "op_type", "") in ("Input", "Output"):
                            continue
                        inlined_id = f"{node_id}_pruned_{b_id}"
                        id_map[b_id] = inlined_id
                        inlined_inputs = [id_map.get(inp, inp) for inp in b_node.inputs]
                        cloned = clone_logical_node(b_node, id=inlined_id, inputs=inlined_inputs)
                        new_nodes[inlined_id] = IRNode(**cloned.__dict__)

                    # Point conditional node to branch output
                    b_outs = getattr(active_branch, "outputs", [])
                    out_ref = id_map.get(b_outs[0], b_outs[0]) if b_outs else list(new_nodes.keys())[-1]
                    identity = clone_logical_node(node, id=node_id, op_type="Identity", inputs=[out_ref])
                    new_nodes[node_id] = IRNode(**identity.__dict__)
                    continue

            # Step 2: Invariant hoisting if predicate is not statically evaluable
            if true_branch and false_branch:
                hoisted = _hoist_common_invariants(graph, node_id, true_branch, false_branch)
                if hoisted:
                    modified = True
                    for h_node in hoisted:
                        new_nodes[h_node.id] = h_node

        new_nodes[node_id] = node if isinstance(node, IRNode) else (IRNode(**node.__dict__) if hasattr(node, "__dict__") else node)

    if modified:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return graph
