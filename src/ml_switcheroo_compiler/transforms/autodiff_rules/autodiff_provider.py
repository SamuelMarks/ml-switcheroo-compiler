"""Autodiff Provider for Data-Driven Rules."""

from __future__ import annotations

import logging
import re
import typing
from collections.abc import Callable

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, as_opt_graph
from ml_switcheroo_compiler.ops.base import emit_ir_node

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.grad.config_models import AutodiffRulesManifestModel


@typing.runtime_checkable
class _ShapeMetadataNodeProtocol(typing.Protocol):
    """Protocol for graph nodes exposing optional `shape_metadata`."""

    shape_metadata: tuple[int | str, ...] | list[int | str] | None


@typing.runtime_checkable
class _IdNodeProtocol(typing.Protocol):
    """Protocol for graph nodes exposing an `id` attribute."""

    id: str


def _get_node_shape_meta(node: IRNode | TensorOrScalar | None) -> tuple[int | str, ...] | list[int | str] | None:
    """Safely extract `shape_metadata` from a node or return None.

    Args:
        node (IRNode | TensorOrScalar | None): Candidate node object.

    Returns:
        tuple[int | str, ...] | list[int | str] | None: Shape metadata if present, else None.
    """
    return node.shape_metadata if isinstance(node, _ShapeMetadataNodeProtocol) else None


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
    graph: IRGraph,
    args: list[str],
    node: IRNode,
    cotangent: str | None = None,
    tangents: list[str] | None = None,
) -> tuple[list[str], dict[str, AttributeValue]]:
    """Parse positional input expressions and keyword attribute definitions.

    Args:
        graph (IRGraph): The computation graph.
        args (list[str]): List of argument strings.
        node (IRNode): The primal node.
        cotangent (Optional[str]): Cotangent identifier.
        tangents (Optional[list[str]]): Tangent identifiers.

    Returns:
        tuple[list[str], dict[str, AttributeValue]]: Parsed input identifiers and keyword attributes.
    """
    parsed_inputs: list[str] = []
    parsed_attrs: dict[str, AttributeValue] = {}
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
    graph: IRGraph,
    op: str,
    args: list[str],
    node: IRNode,
    cotangent: str | None = None,
    tangents: list[str] | None = None,
) -> str:
    """Build an IR node for an invocation parsed from derivative expression rules.

    Args:
        graph (IRGraph): The computation graph.
        op (str): Name of the operator to construct.
        args (list[str]): Raw string arguments.
        node (IRNode): The primal IR node.
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

            new_node = LogicalNode(id=node_id, op_type="Constant", attributes={"value": val}, shape_metadata=_get_node_shape_meta(node))
            graph.nodes[node_id] = new_node
        return node_id

    if op in ("BroadcastAlign", "BroadcastReduce"):
        cot_id, tgt_id = parsed_inputs[0], parsed_inputs[1]
        shape_meta = _get_node_shape_meta(graph.nodes.get(tgt_id)) if tgt_id in graph.nodes else None
        return emit_ir_node(graph, "BroadcastReduce", [cot_id, tgt_id], shape_metadata=shape_meta, attributes={"target_id": tgt_id})

    if op == "BroadcastLike":
        src_id, tgt_id = parsed_inputs[0], parsed_inputs[1]
        shape_meta = _get_node_shape_meta(graph.nodes.get(tgt_id)) if tgt_id in graph.nodes else None
        return emit_ir_node(graph, "BroadcastLike", [src_id, tgt_id], shape_metadata=shape_meta, attributes={"target_id": tgt_id})

    attrs = dict(node.attributes) if op == node.op_type else {}
    if not attrs and op == "SetItem":
        attrs = dict(node.attributes)
    if not attrs:
        if (op == "ExpandDims" and node.op_type == "Squeeze") or (op == "Squeeze" and node.op_type == "ExpandDims"):
            attrs = dict(node.attributes)
    attrs.update(parsed_attrs)

    shape_meta = _get_node_shape_meta(node)
    if op == "Reshape" and node.inputs and node.inputs[0] in graph.nodes:
        shape_meta = _get_node_shape_meta(graph.nodes[node.inputs[0]])
    return emit_ir_node(graph, op, parsed_inputs, shape_meta, attributes=attrs or None)


def _resolve_variable_ref(
    expr: str,
    node: IRNode,
    cotangent: str | None,
    tangents: list[str] | None,
) -> str:
    """Resolve a variable placeholder or literal reference in an autodiff rule expression.

    Args:
        expr (str): Stripped variable reference string.
        node (IRNode): The primal IR node being differentiated.
        cotangent (str | None): The output cotangent ID for VJP.
        tangents (list[str] | None): The input tangent IDs for JVP.

    Returns:
        str: Resolved node identifier or literal string.
    """
    if expr == "$cotangent":
        assert cotangent is not None
        return cotangent
    if expr == "$output":
        raw_node: TensorOrScalar = node
        return str(raw_node.id) if isinstance(raw_node, _IdNodeProtocol) else ""
    if expr.startswith("$input["):
        match = re.match(r"\$input\[(\d+)\]", expr)
        assert match is not None
        return str(node.inputs[int(match.group(1))])
    if expr.startswith("$tangent["):
        assert tangents is not None
        match = re.match(r"\$tangent\[(\d+)\]", expr)
        assert match is not None
        return tangents[int(match.group(1))]
    return expr


def _parse_expression(
    graph: IRGraph,
    expr: str,
    node: IRNode,
    cotangent: str | None = None,
    tangents: list[str] | None = None,
) -> str:
    """Parse a string expression into IR nodes.

    Args:
        graph (IRGraph): The computation graph being constructed.
        expr (str): String representation of the derivative expression.
        node (IRNode): The primal IR node being differentiated.
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
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    if expr.startswith("- ") or (expr.startswith("-") and not expr.startswith("-=")):
        inner_str = expr[2:].strip() if expr.startswith("- ") else expr[1:].strip()
        inner_id = _parse_expression(graph, inner_str, node, cotangent, tangents)
        return emit_ir_node(graph, "Negative", [inner_id], _get_node_shape_meta(node))

    m = re.match(r"^([A-Za-z0-9_]+)\((.*)\)$", expr)
    if m:
        op = m.group(1)
        args = _split_nested_args(m.group(2))
        return _build_parsed_call(graph, op, args, node, cotangent, tangents)

    return _resolve_variable_ref(expr, node, cotangent, tangents)


def _is_ir_node(val: TensorOrScalar) -> typing.TypeGuard[IRNode]:
    """Type guard narrowing `val` to `IRNode`.

    Args:
        val (TensorOrScalar): Candidate IR node value.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def _as_ir_node(val: TensorOrScalar) -> IRNode:
    """Narrow a runtime node argument to `IRNode`.

    Args:
        val (TensorOrScalar): Candidate IR node value.

    Returns:
        IRNode: Narrowed IRNode instance.
    """
    assert _is_ir_node(val)
    return val


def _is_ir_graph(val: TensorOrScalar) -> typing.TypeGuard[IRGraph]:
    """Type guard narrowing `val` to `IRGraph`.

    Args:
        val (TensorOrScalar): Candidate IR graph value.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def _ensure_ir_graph(val: TensorOrScalar) -> IRGraph:
    """Narrow a runtime graph argument to `IRGraph`.

    Args:
        val (TensorOrScalar): Candidate IR graph value.

    Returns:
        IRGraph: Narrowed IRGraph instance.
    """
    assert _is_ir_graph(val)
    return val


def _fallback_finite_difference_jvp(graph: TensorOrScalar, node: TensorOrScalar, tangents: TensorOrScalar) -> str:
    """Implement a generic finite difference fallback for JVP.

    Args:
        graph (TensorOrScalar): Input `graph` argument (graph).
        node (TensorOrScalar): Input `node` argument (node).
        tangents (TensorOrScalar): Input `tangents` argument (tangents).

    Returns:
        str: Emitted finite-difference JVP node ID.
    """
    # JVP ~ (f(x + epsilon * t) - f(x - epsilon * t)) / (2 * epsilon)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    ir_graph = _ensure_ir_graph(graph)
    ir_node = _as_ir_node(node)
    opt_graph = as_opt_graph(ir_graph)
    node_shape = _get_node_shape_meta(ir_node)

    epsilon = emit_ir_node(opt_graph, "Constant", [], None, attributes={"value": 0.0001})
    two_eps = emit_ir_node(opt_graph, "Constant", [], None, attributes={"value": 0.0002})

    pos_inputs: list[str] = []
    neg_inputs: list[str] = []

    tangents_list: list[str] = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
    for _idx, (inp, tang) in enumerate(zip(ir_node.inputs, tangents_list)):
        inp_shape = _get_node_shape_meta(ir_graph.nodes.get(inp))
        eps_t = emit_ir_node(opt_graph, "Multiply", [epsilon, tang], inp_shape)
        pos_inputs.append(emit_ir_node(opt_graph, "Add", [inp, eps_t], inp_shape))
        neg_inputs.append(emit_ir_node(opt_graph, "Subtract", [inp, eps_t], inp_shape))

    for i in range(len(tangents_list), len(ir_node.inputs)):
        pos_inputs.append(ir_node.inputs[i])
        neg_inputs.append(ir_node.inputs[i])

    f_pos = emit_ir_node(opt_graph, ir_node.op_type, pos_inputs, node_shape, attributes=ir_node.attributes)
    f_neg = emit_ir_node(opt_graph, ir_node.op_type, neg_inputs, node_shape, attributes=ir_node.attributes)

    diff = emit_ir_node(opt_graph, "Subtract", [f_pos, f_neg], node_shape)
    return emit_ir_node(opt_graph, "Divide", [diff, two_eps], node_shape)


_MANIFEST_MODEL_CACHE: AutodiffRulesManifestModel | None = None
_MODULAR_RULES_CACHE: dict[str, dict[str, AttributeValue]] | None = None


def _get_autodiff_manifest() -> AutodiffRulesManifestModel | None:
    """Retrieve cached AutodiffRulesManifestModel parsed from modular autodiff rules.

    Returns:
        Optional[AutodiffRulesManifestModel]: Validated AutodiffRulesManifestModel instance or None.
    """
    global _MANIFEST_MODEL_CACHE
    if _MANIFEST_MODEL_CACHE is not None:
        return _MANIFEST_MODEL_CACHE

    import os

    import yaml

    from ml_switcheroo_compiler.grad.config_models import AutodiffRulesManifestModel

    rules_dir = os.path.dirname(__file__)
    jvp_path = os.path.join(rules_dir, "jvp_rules.yaml")
    vjp_path = os.path.join(rules_dir, "vjp_rules.yaml")
    if os.path.exists(jvp_path) and os.path.exists(vjp_path):
        try:
            with open(jvp_path, encoding="utf-8") as f:
                jvp_data = yaml.safe_load(f) or {}
            with open(vjp_path, encoding="utf-8") as f:
                vjp_data = yaml.safe_load(f) or {}
            _MANIFEST_MODEL_CACHE = AutodiffRulesManifestModel.model_validate({"jvp_rules": jvp_data, "vjp_rules": vjp_data})
            return _MANIFEST_MODEL_CACHE
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return None


def _get_modular_rules() -> dict[str, dict[str, AttributeValue]]:
    """Load and cache modular derivative YAML schemas.

    Returns:
        dict[str, dict[str, AttributeValue]]: Dictionary containing parsed jvp_rules and vjp_rules.
    """
    global _MODULAR_RULES_CACHE
    if _MODULAR_RULES_CACHE is not None:
        return _MODULAR_RULES_CACHE

    import os

    import yaml

    combined: dict[str, dict[str, AttributeValue]] = {"jvp_rules": {}, "vjp_rules": {}}
    dir_path = os.path.dirname(__file__)
    for filename in (
        "linalg_derivatives.yaml",
        "nn_derivatives.yaml",
        "reduction_derivatives.yaml",
        "math_derivatives.yaml",
    ):
        file_path = os.path.join(dir_path, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                if isinstance(content, dict):
                    if "jvp_rules" in content and isinstance(content["jvp_rules"], dict):
                        combined["jvp_rules"].update(content["jvp_rules"])
                    if "vjp_rules" in content and isinstance(content["vjp_rules"], dict):
                        combined["vjp_rules"].update(content["vjp_rules"])
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    _MODULAR_RULES_CACHE = combined
    return combined


def _lookup_manifest_rule(op_type: str, rule_type: str) -> dict[str, AttributeValue] | None:
    """Look up autodiff rule in the typed manifest model.

    Args:
        op_type (str): Name of the operation.
        rule_type (str): 'vjp' or 'jvp'.

    Returns:
        dict[str, AttributeValue] | None: Rule dictionary if found, else None.
    """
    manifest = _get_autodiff_manifest()
    if manifest is None:
        return None
    rules_dict = manifest.vjp_rules if rule_type == "vjp" else manifest.jvp_rules
    for candidate in (op_type, op_type.capitalize(), op_type.lower()):
        if candidate in rules_dict:
            rule_obj = rules_dict[candidate]
            if rule_type == "vjp" and rule_obj.vjp:
                return {"vjp": rule_obj.vjp if isinstance(rule_obj.vjp, list) else [rule_obj.vjp]}
            if rule_type == "jvp" and rule_obj.jvp:
                return {"jvp": rule_obj.jvp}
    return None


def _lookup_modular_rule(op_type: str, rule_type: str) -> dict[str, AttributeValue] | None:
    """Look up autodiff rule in modular derivative YAML caches.

    Args:
        op_type (str): Name of the operation.
        rule_type (str): 'vjp' or 'jvp'.

    Returns:
        dict[str, AttributeValue] | None: Rule dictionary if found, else None.
    """
    modular = _get_modular_rules()
    target_section = "vjp_rules" if rule_type == "vjp" else "jvp_rules"
    section_map = modular.get(target_section, {})
    for candidate in (op_type, op_type.capitalize(), op_type.lower()):
        entry = section_map.get(candidate)
        if isinstance(entry, dict):
            return entry
    return None


def _lookup_primitive_registry_rule(op_type: str, rule_type: str) -> dict[str, AttributeValue] | None:
    """Look up autodiff rule in primitive VJP/JVP registries or OPS_REGISTRY.

    Args:
        op_type (str): Name of the operation.
        rule_type (str): 'vjp' or 'jvp'.

    Returns:
        dict[str, AttributeValue] | None: Rule dictionary if found, else None.
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
    return None


def _load_autodiff_rule(op_type: str, rule_type: str) -> dict[str, AttributeValue] | None:
    """Load autodiff rule from typed manifest model, modular files, registry or YAML files.

    Args:
        op_type (str): Name of the operation.
        rule_type (str): 'vjp' or 'jvp'.

    Returns:
        Optional[dict[str, AttributeValue]]: Rule dictionary if found, else None.
    """
    return _lookup_manifest_rule(op_type, rule_type) or _lookup_modular_rule(op_type, rule_type) or _lookup_primitive_registry_rule(op_type, rule_type) or _load_rule_from_yaml_files(op_type, rule_type)


def _read_yaml_dict(file_path: str) -> dict[str, AttributeValue]:
    """Read a YAML file and return its contents if it is a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict[str, AttributeValue]: Parsed dictionary or empty dict if not found or invalid.
    """
    import os

    import yaml

    if not os.path.exists(file_path):
        return {}
    with open(file_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _find_candidate_in_dict(
    data: dict[str, AttributeValue],
    candidates: tuple[str, ...],
) -> dict[str, AttributeValue] | None:
    """Find the first matching candidate key in a dictionary.

    Args:
        data (dict[str, AttributeValue]): Dictionary to search.
        candidates (tuple[str, ...]): Ordered candidate keys.

    Returns:
        dict[str, AttributeValue] | None: Matched rule dictionary or None.
    """
    for name in candidates:
        val = data.get(name)
        if isinstance(val, dict):
            return {str(k): v for k, v in val.items()}
    return None


def _load_rule_from_yaml_files(op_type: str, rule_type: str) -> dict[str, AttributeValue] | None:
    """Load autodiff rule from fallback filesystem YAML files.

    Args:
        op_type (str): Name of the operation.
        rule_type (str): 'vjp' or 'jvp'.

    Returns:
        Optional[dict[str, AttributeValue]]: Rule dictionary if found, else None.
    """
    import os

    base_dir = os.path.dirname(__file__)
    candidates = (op_type, op_type.lower(), op_type.capitalize())

    rule_data = _read_yaml_dict(os.path.join(base_dir, f"{rule_type}_rules.yaml"))
    found = _find_candidate_in_dict(rule_data, candidates)
    if found is not None:
        return found

    for candidate in candidates:
        cand_data = _read_yaml_dict(os.path.join(base_dir, "rules", f"{candidate}.yaml"))
        found = _find_candidate_in_dict(cand_data, candidates)
        if found is not None:
            return found

    return None


def get_vjp_from_data(op_type: str) -> Callable[[IRGraph, IRNode, str], tuple[str, ...]] | None:
    """Get VJP handler from data-driven rules.

    Args:
        op_type (str): Operation identifier.

    Returns:
        Callable[[IRGraph, IRNode, str], tuple[str, ...]] | None: Evaluator callable or None.
    """
    ad_rules = _load_autodiff_rule(op_type, "vjp")
    if not ad_rules or "vjp" not in ad_rules or not ad_rules["vjp"]:
        return None

    vjp_exprs = ad_rules["vjp"]
    vjp_exprs_list: list[str] = [str(e) for e in vjp_exprs] if isinstance(vjp_exprs, (list, tuple)) else [str(vjp_exprs)]

    def data_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str, ...]:
        """Execute data vjp for compiler processing.

        Args:
            graph (IRGraph): Input `graph` argument (graph).
            node (IRNode): Input `node` argument (node).
            cotangent (str): Input `cotangent` argument (cotangent).

        Returns:
            tuple[str, ...]: Computed `tuple[str, ...]` output.
        """
        adjs = [_parse_expression(graph, expr, node, cotangent=cotangent) for expr in vjp_exprs_list]
        return tuple(adjs)

    return data_vjp


def get_jvp_from_data(op_type: str) -> Callable[[TensorOrScalar, TensorOrScalar, TensorOrScalar], str] | None:
    """Get JVP handler from data-driven rules.

    Args:
        op_type (str): Operation identifier.

    Returns:
        Callable[[TensorOrScalar, TensorOrScalar, TensorOrScalar], str] | None: Evaluator callable.
    """
    ad_rules = _load_autodiff_rule(op_type, "jvp")
    if not ad_rules or "jvp" not in ad_rules or not ad_rules["jvp"]:
        return _fallback_finite_difference_jvp

    jvp_expr = ad_rules["jvp"]

    def data_jvp(graph: TensorOrScalar, node: TensorOrScalar, tangents: TensorOrScalar) -> str:
        """Evaluate JVP expression for `node`.

        Args:
            graph (TensorOrScalar): Input `graph` argument (graph).
            node (TensorOrScalar): Input `node` argument (node).
            tangents (TensorOrScalar): Input `tangents` argument (tangents).

        Returns:
            str: Emitted JVP tangent node ID.
        """
        tangents_list: list[str] = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
        expr_str = jvp_expr if isinstance(jvp_expr, str) else str(jvp_expr)
        return _parse_expression(_ensure_ir_graph(graph), expr_str, _as_ir_node(node), tangents=tangents_list)

    return data_jvp
