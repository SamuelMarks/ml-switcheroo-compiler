# ruff: noqa: C901, PLR0911, PLR0912
"""Autodiff Provider for Data-Driven Rules."""

import re
from typing import Any, Optional

from ml_switcheroo_compiler.ops.base import emit_ir_node


def _split_nested_args(args_str: str) -> list[str]:
    """Split comma-separated arguments while respecting nested parentheses and brackets.

    Args:
        args_str (str): The arguments string to partition.

    Returns:
        list[str]: Extracted individual argument substrings.
    """
    args: list[str] = []
    depth = 0
    current_arg = ""
    for char in args_str:
        if char in ("(", "[", "{"):
            depth += 1
        elif char in (")", "]", "}"):
            depth -= 1
        elif char == "," and depth == 0:
            args.append(current_arg.strip())
            current_arg = ""
            continue
        current_arg += char
    if current_arg:
        args.append(current_arg.strip())
    return args


def _parse_args_and_attrs(
    graph: Any,
    args: list[str],
    node: Any,
    cotangent: Optional[str] = None,
    tangents: Optional[list[str]] = None,
) -> tuple[list[str], dict[str, Any]]:
    """Parse positional input expressions and keyword attribute definitions.

    Args:
        graph (Any): The computation graph.
        args (list[str]): List of argument strings.
        node (Any): The primal node.
        cotangent (Optional[str]): Cotangent identifier.
        tangents (Optional[list[str]]): Tangent identifiers.

    Returns:
        tuple[list[str], dict[str, Any]]: Parsed input identifiers and keyword attributes.
    """
    parsed_inputs: list[str] = []
    parsed_attrs: dict[str, Any] = {}
    for arg in args:
        if "=" in arg and not arg.startswith("("):
            k, v = arg.split("=", 1)
            import ast

            try:
                parsed_attrs[k.strip()] = ast.literal_eval(v.strip())
            except Exception:
                parsed_attrs[k.strip()] = v.strip()
        else:
            parsed_inputs.append(_parse_expression(graph, arg, node, cotangent, tangents))
    return parsed_inputs, parsed_attrs


def _build_parsed_call(
    graph: Any,
    op: str,
    args: list[str],
    node: Any,
    cotangent: Optional[str] = None,
    tangents: Optional[list[str]] = None,
) -> str:
    """Build an IR node for an invocation parsed from derivative expression rules.

    Args:
        graph (Any): The computation graph.
        op (str): Name of the operator to construct.
        args (list[str]): Raw string arguments.
        node (Any): The primal IR node.
        cotangent (Optional[str]): The cotangent identifier.
        tangents (Optional[list[str]]): Tangent identifiers.

    Returns:
        str: Identifier of the created IR node.
    """
    parsed_inputs, parsed_attrs = _parse_args_and_attrs(graph, args, node, cotangent, tangents)

    alias_map = {
        "Neg": "Negative",
        "Sub": "Subtract",
        "Mul": "Multiply",
        "Div": "TrueDivide",
        "Divide": "TrueDivide",
    }
    op = alias_map.get(op, op)

    if op == "Constant":
        val = float(parsed_inputs[0]) if parsed_inputs else 0.0
        node_id = f"cst_ad_{id(node)}_{val}".replace(".", "_").replace("-", "neg_")
        if node_id not in graph.nodes:
            from ml_switcheroo_compiler.ir.core import LogicalNode

            new_node = LogicalNode(id=node_id, op_type="Constant", attributes={"value": val}, shape_metadata=getattr(node, "shape_metadata", None))
            graph.nodes[node_id] = new_node
        return node_id

    if op in ("BroadcastAlign", "BroadcastReduce"):
        cot_id, tgt_id = parsed_inputs[0], parsed_inputs[1]
        shape_meta = getattr(graph.nodes.get(tgt_id), "shape_metadata", None) if hasattr(graph, "nodes") and tgt_id in graph.nodes else None
        return emit_ir_node(graph, "BroadcastReduce", [cot_id, tgt_id], shape_metadata=shape_meta, attributes={"target_id": tgt_id})

    if op == "BroadcastLike":
        src_id, tgt_id = parsed_inputs[0], parsed_inputs[1]
        shape_meta = getattr(graph.nodes.get(tgt_id), "shape_metadata", None) if hasattr(graph, "nodes") and tgt_id in graph.nodes else None
        return emit_ir_node(graph, "BroadcastLike", [src_id, tgt_id], shape_metadata=shape_meta, attributes={"target_id": tgt_id})

    attrs = dict(node.attributes) if hasattr(node, "attributes") and op == getattr(node, "op_type", "") else {}
    if not attrs and op == "SetItem" and hasattr(node, "attributes"):
        attrs = dict(node.attributes)
    if not attrs and hasattr(node, "attributes"):
        if (op == "ExpandDims" and getattr(node, "op_type", "") == "Squeeze") or (op == "Squeeze" and getattr(node, "op_type", "") == "ExpandDims"):
            attrs = dict(node.attributes)
    attrs.update(parsed_attrs)

    shape_meta = getattr(node, "shape_metadata", None)
    if op == "Reshape" and hasattr(node, "inputs") and node.inputs and hasattr(graph, "nodes") and node.inputs[0] in graph.nodes:
        shape_meta = getattr(graph.nodes[node.inputs[0]], "shape_metadata", None)
    return emit_ir_node(graph, op, parsed_inputs, shape_meta, attributes=attrs or None)


def _parse_expression(
    graph: Any,
    expr: str,
    node: Any,
    cotangent: Optional[str] = None,
    tangents: Optional[list[str]] = None,
) -> str:
    """Parse a string expression into IR nodes.

    Args:
        graph (Any): The computation graph being constructed.
        expr (str): String representation of the derivative expression.
        node (Any): The primal IR node being differentiated.
        cotangent (Optional[str]): The output cotangent ID for VJP.
        tangents (Optional[list[str]]): The input tangent IDs for JVP.

    Returns:
        str: Emitted IR node identifier representing the evaluated expression.
    """
    expr = expr.strip()
    try:
        float(expr)
        return expr
    except ValueError:
        pass

    if expr.startswith("- ") or (expr.startswith("-") and not expr.startswith("-=")):
        inner_str = expr[2:].strip() if expr.startswith("- ") else expr[1:].strip()
        inner_id = _parse_expression(graph, inner_str, node, cotangent, tangents)
        return emit_ir_node(graph, "Negative", [inner_id], getattr(node, "shape_metadata", None))

    if expr == "$cotangent":
        assert cotangent is not None
        return cotangent

    if expr == "$output":
        return getattr(node, "id", "")

    m = re.match(r"^([A-Za-z0-9_]+)\((.*)\)$", expr)
    if m:
        op = m.group(1)
        args = _split_nested_args(m.group(2))
        return _build_parsed_call(graph, op, args, node, cotangent, tangents)

    if expr.startswith("$input["):
        idx = int(re.match(r"\$input\[(\d+)\]", expr).group(1))
        return node.inputs[idx]
    if expr.startswith("$tangent["):
        assert tangents is not None
        idx = int(re.match(r"\$tangent\[(\d+)\]", expr).group(1))
        return tangents[idx]

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


def _load_autodiff_rule(op_type: str, rule_type: str) -> Optional[dict[str, Any]]:
    """Load autodiff rule from registry or YAML files.

    Args:
        op_type (str): Name of the operation.
        rule_type (str): 'vjp' or 'jvp'.

    Returns:
        Optional[dict[str, Any]]: Rule dictionary if found, else None.
    """
    if rule_type == "vjp":
        from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import load_primitive_vjp_rules

        vjp_rules = load_primitive_vjp_rules()
        for candidate in (op_type, op_type.capitalize(), op_type.lower()):
            if candidate in vjp_rules:
                return {"vjp": vjp_rules[candidate].vjp}
    elif rule_type == "jvp":
        from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import load_primitive_jvp_rules

        jvp_rules = load_primitive_jvp_rules()
        for candidate in (op_type, op_type.capitalize(), op_type.lower()):
            if candidate in jvp_rules:
                return {"jvp": jvp_rules[candidate].jvp}

    from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

    op_def = OPS_REGISTRY.get(op_type, {})
    if isinstance(op_def, dict):
        ad_rules = op_def.get("autodiff", {})
        if isinstance(ad_rules, dict) and rule_type in ad_rules:
            return ad_rules

    import os

    import yaml

    rule_file = os.path.join(os.path.dirname(__file__), f"{rule_type}_rules.yaml")
    if os.path.exists(rule_file):
        with open(rule_file) as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and op_type in data:
                return data[op_type]

    yaml_path = os.path.join(os.path.dirname(__file__), "rules", f"{op_type}.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and op_type in data:
                return data[op_type]

    legacy_path = os.path.join(os.path.dirname(__file__), "autodiff_rules.yaml")
    if os.path.exists(legacy_path):
        with open(legacy_path) as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and op_type in data:
                return data[op_type]

    return None


def get_vjp_from_data(op_type: str) -> Any:
    """Get VJP handler from data-driven rules.

    Args:
        op_type (str): Operation identifier.

    Returns:
        Any: Evaluator callable or None.
    """
    ad_rules = _load_autodiff_rule(op_type, "vjp")
    if not ad_rules or "vjp" not in ad_rules:
        return None

    vjp_exprs = ad_rules["vjp"]

    def data_vjp(graph: Any, node: Any, cotangent: str) -> Any:
        """data_vjp function.

        Args:
            graph (object): The graph parameter.
            node (object): The node parameter.
            cotangent (object): The cotangent parameter.

        Returns:
            object: Result.
        """
        adjs = [_parse_expression(graph, expr, node, cotangent=cotangent) for expr in vjp_exprs]
        return tuple(adjs)

    return data_vjp


def get_jvp_from_data(op_type: str) -> Any:
    """Get JVP handler from data-driven rules.

    Args:
        op_type (str): Operation identifier.

    Returns:
        Any: Evaluator callable.
    """
    ad_rules = _load_autodiff_rule(op_type, "jvp")
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
