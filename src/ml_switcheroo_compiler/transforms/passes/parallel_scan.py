# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Parallel scan transformation pass converting sequential scan into parallel scan trees."""

from __future__ import annotations

import os
from typing import Any, cast

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node


def _load_scan_rules() -> dict[str, Any]:
    """Load declarative parallel scan configuration from YAML.

    Returns:
        dict[str, Any]: The loaded parallel scan rules.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "parallel_scan_rules.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
    return {}


def detect_associative_reduction(body_graph: Any, rules: dict[str, Any]) -> tuple[str, str] | None:
    """Detect whether a scan body graph performs an associative reduction.

    Args:
        body_graph (Any): The scan body graph or block.
        rules (dict[str, Any]): Declarative associative scan rules.

    Returns:
        Optional[tuple[str, str]]: Tuple of (detected_op_type, parallel_primitive) if associative, else None.
    """
    if not body_graph:
        return None

    nodes = body_graph.nodes if isinstance(body_graph.nodes, dict) else {n.id: n for n in getattr(body_graph, "nodes", [])}
    assoc_map = rules.get("associative_ops", {})

    # Check non-input/output operational nodes
    compute_nodes = [n for n in nodes.values() if getattr(n, "op_type", "") not in ("Input", "Output")]
    if len(compute_nodes) == 1:
        op_node = compute_nodes[0]
        op_type = getattr(op_node, "op_type", "")
        if op_type in assoc_map:
            prim = assoc_map[op_type].get("parallel_primitive", "CumSum")
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
        if getattr(node, "op_type", "") == "Scan":
            body = node.attributes.get("body")
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
                    shape_metadata=getattr(node, "shape_metadata", ()),
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
                        shape_metadata=getattr(node, "shape_metadata", ()),
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
                new_nodes[node_id] = IRNode(**identity.__dict__)
                continue

        new_nodes[node_id] = node if isinstance(node, IRNode) else IRNode(**node.__dict__)

    if transformed:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)

    return graph
