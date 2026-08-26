"""Autodiff Provider for Data-Driven Rules."""

import re
from typing import Optional

from ml_switcheroo_compiler.ops.base import emit_ir_node


def _parse_expression(graph, expr: str, node, cotangent: Optional[str] = None, tangents: Optional[list[str]] = None) -> str:  # noqa: C901, PLR0912
    """Parse a string expression into IR nodes."""
    if expr == "$cotangent":
        assert cotangent is not None
        return cotangent

    # Very basic recursive parser for expressions like OpName(arg1, arg2)
    # We find the outermost call
    m = re.match(r"^([A-Za-z0-9_]+)\((.*)\)$", expr.strip())
    if m:
        op = m.group(1)
        args_str = m.group(2)

        # Split by comma, respecting nested parens
        args = []
        depth = 0
        current_arg = ""
        for char in args_str:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
                continue
            current_arg += char
        if current_arg:
            args.append(current_arg.strip())

        evaluated_args = []
        for arg in args:
            evaluated_args.append(_parse_expression(graph, arg, node, cotangent, tangents))

        if op == "Constant":
            # Evaluated args[0] is just a string like "1.0"
            val = float(args[0])
            node_id = f"cst_ad_{id(node)}_{val}".replace(".", "_")
            if node_id not in graph.nodes:
                from ml_switcheroo_compiler.ir.core import LogicalNode

                new_node = LogicalNode(id=node_id, op_type="Constant", attributes={"value": val}, shape_metadata=getattr(node, "shape_metadata", None))
                graph.nodes[node_id] = new_node
            return node_id

        attrs = node.attributes if hasattr(node, "attributes") and op == getattr(node, "op_type", "") else None
        if not attrs and op == "SetItem":
            attrs = node.attributes
        return emit_ir_node(graph, op, evaluated_args, getattr(node, "shape_metadata", None), attributes=attrs)

    elif expr.startswith("$input["):
        idx = int(re.match(r"\$input\[(\d+)\]", expr).group(1))
        return node.inputs[idx]
    elif expr.startswith("$tangent["):
        assert tangents is not None
        idx = int(re.match(r"\$tangent\[(\d+)\]", expr).group(1))
        return tangents[idx]

    # If it's just a raw variable name fallback
    return expr


def _fallback_finite_difference_jvp(graph, node, tangents) -> str:
    """Implement a generic finite difference fallback for JVP."""
    # JVP ~ (f(x + epsilon * t) - f(x - epsilon * t)) / (2 * epsilon)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    epsilon = emit_ir_node(graph, "Constant", [], None, attributes={"value": 1e-4})
    two_eps = emit_ir_node(graph, "Constant", [], None, attributes={"value": 2e-4})

    pos_inputs = []
    neg_inputs = []

    for _idx, (inp, tang) in enumerate(zip(node.inputs, tangents)):
        eps_t = emit_ir_node(graph, "Multiply", [epsilon, tang], getattr(graph.nodes.get(inp), "shape_metadata", None))
        pos_inputs.append(emit_ir_node(graph, "Add", [inp, eps_t], getattr(graph.nodes.get(inp), "shape_metadata", None)))
        neg_inputs.append(emit_ir_node(graph, "Subtract", [inp, eps_t], getattr(graph.nodes.get(inp), "shape_metadata", None)))

    for i in range(len(tangents), len(node.inputs)):
        pos_inputs.append(node.inputs[i])
        neg_inputs.append(node.inputs[i])

    f_pos = emit_ir_node(graph, node.op_type, pos_inputs, getattr(node, "shape_metadata", None), attributes=node.attributes)
    f_neg = emit_ir_node(graph, node.op_type, neg_inputs, getattr(node, "shape_metadata", None), attributes=node.attributes)

    diff = emit_ir_node(graph, "Subtract", [f_pos, f_neg], getattr(node, "shape_metadata", None))
    return emit_ir_node(graph, "Divide", [diff, two_eps], getattr(node, "shape_metadata", None))


def get_vjp_from_data(op_type: str):
    """Get vjp."""
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY

    op_def = _YAML_REGISTRY.get(op_type, {})
    ad_rules = op_def.get("autodiff", {})
    if not ad_rules or "vjp" not in ad_rules:
        return None

    vjp_exprs = ad_rules["vjp"]

    def data_vjp(graph, node, cotangent: str):
        """data_vjp function.

        Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (object): The cotangent parameter.

        Returns:
        object: Result.
        """
        adjs = []
        for expr in vjp_exprs:
            adjs.append(_parse_expression(graph, expr, node, cotangent=cotangent))
        return tuple(adjs)

    return data_vjp


def get_jvp_from_data(op_type: str):
    """Get jvp."""
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY

    op_def = _YAML_REGISTRY.get(op_type, {})
    ad_rules = op_def.get("autodiff", {})
    if not ad_rules or "jvp" not in ad_rules:
        return _fallback_finite_difference_jvp

    jvp_expr = ad_rules["jvp"]

    def data_jvp(graph, node, tangents) -> str:
        """data_jvp function.

        Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (object): The tangents parameter.

        Returns:
        object: Result.
        """
        tangents_list = list(tangents) if isinstance(tangents, (tuple, list)) else [tangents]
        return _parse_expression(graph, jvp_expr, node, tangents=tangents_list)

    return data_jvp
