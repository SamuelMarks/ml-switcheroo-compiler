# ruff: noqa: E402
"""Parallel scan transformation pass converting sequential scan into parallel scan trees."""

from __future__ import annotations

import os

import yaml
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.core.shape import _get_attr, _has_attr, _obj_vars
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, as_opt_graph, clone_logical_node


def _load_scan_rules() -> dict[str, AttributeValue]:
    """Load declarative parallel scan configuration from YAML.

    Returns:
        dict[str, AttributeValue]: The loaded parallel scan rules.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "parallel_scan_rules.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
    return {}


def detect_associative_reduction(body_graph: IRGraph | None, rules: dict[str, AttributeValue]) -> tuple[str, str] | None:
    """Detect whether a scan body graph performs an associative reduction.

    Args:
        body_graph (IRGraph | None): The scan body graph or block.
        rules (dict[str, AttributeValue]): Declarative associative scan rules.

    Returns:
        Optional[tuple[str, str]]: Tuple of (detected_op_type, parallel_primitive) if associative, else None.
    """
    if not body_graph:
        return None

    nodes = body_graph.nodes if _has_attr(body_graph.nodes, "values") else {n.id: n for n in _get_attr(body_graph, "nodes", [])}
    raw_assoc = rules.get("associative_ops", {})
    assoc_map = raw_assoc if isinstance(raw_assoc, dict) else {}

    # Check non-input/output operational nodes
    compute_nodes = [n for n in nodes.values() if _get_attr(n, "op_type", "") not in ("Input", "Output")]
    if len(compute_nodes) == 1:
        op_node = compute_nodes[0]
        op_type = str(_get_attr(op_node, "op_type", ""))
        if op_type in assoc_map:
            entry = assoc_map[op_type]
            raw_prim = entry.get("parallel_primitive", "CumSum") if isinstance(entry, dict) else "CumSum"
            prim = str(raw_prim)
            return op_type, prim

    return None


def parallel_scan_pass(graph: IRGraph) -> IRGraph:
    """Optimize Scan operations by converting sequential dependencies to parallel scan trees.

    Args:
        graph (IRGraph): Input IR graph.

    Returns:
        IRGraph: Transformed IR graph with parallelized scan operations.
    """
    rules = _load_scan_rules()
    new_nodes: dict[str, IRNode] = {}
    transformed = False

    for node_id, node in graph.nodes.items():
        if _get_attr(node, "op_type", "") == "Scan":
            body = as_opt_graph(node.attributes.get("body"))
            reduction_info = detect_associative_reduction(body, rules)

            if reduction_info is not None:
                op_type, parallel_prim = reduction_info
                transformed = True

                # Inputs to Scan are typically [carry_init, xs]
                init_inp = node.inputs[0] if len(node.inputs) > 1 else None
                xs_inp = node.inputs[1] if len(node.inputs) > 1 else node.inputs[0]

                # Create parallel scan primitive node
                prim_id = f"{node_id}_parallel_{parallel_prim.lower()}"
                prim_node = IRNode(
                    id=prim_id,
                    op_type=parallel_prim,
                    inputs=[xs_inp],
                    attributes={"axis": 0},
                    shape_metadata=_get_attr(node, "shape_metadata", ()),
                )
                new_nodes[prim_id] = prim_node

                if init_inp is not None:
                    # If initial carry is present, combine with cumulative result
                    combined_id = f"{node_id}_accum_init"
                    combined_node = IRNode(
                        id=combined_id,
                        op_type=op_type,
                        inputs=[prim_id, init_inp],
                        attributes={},
                        shape_metadata=_get_attr(node, "shape_metadata", ()),
                    )
                    new_nodes[combined_id] = combined_node
                    last_id = combined_id
                else:
                    last_id = prim_id

                # Replace Scan with Identity pointing to the parallel result
                identity = clone_logical_node(
                    node,
                    id=node_id,
                    op_type="Identity",
                    inputs=[last_id],
                )
                new_nodes[node_id] = IRNode(**_obj_vars(identity))
                continue

        new_nodes[node_id] = node if type(node) is IRNode else IRNode(**_obj_vars(node))

    if transformed:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return graph
