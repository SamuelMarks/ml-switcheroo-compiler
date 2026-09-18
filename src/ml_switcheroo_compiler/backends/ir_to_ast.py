"""IRGraph to AST emission and class reconstruction."""

import os
from typing import Optional

import libcst as cst

from ml_switcheroo_compiler.backends.transpiler_config_models import load_transpiler_config
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

_CONFIG_PATH: str = os.path.join(os.path.dirname(__file__), "transpilation_rules.yaml")
_CONFIG = load_transpiler_config(_CONFIG_PATH)


def _build_attribute_chain(names: list[str]) -> cst.BaseExpression:
    """Build a libcst Attribute chain from a list of identifiers.

    Args:
        names (list[str]): List of module and attribute names in order.

    Returns:
        cst.BaseExpression: Chained CST Attribute expression or Name.
    """
    if not names:
        return cst.Name("empty")
    if len(names) == 1:
        return cst.Name(names[0])
    expr: cst.BaseExpression = cst.Name(names[0])
    for name in names[1:]:
        expr = cst.Attribute(value=expr, attr=cst.Name(name))
    return expr


def _emit_call_args(
    node: IRNode,
    target_framework: str,
    var_map: Optional[dict[str, str]] = None,
) -> list[cst.Arg]:
    """Emit CST argument list for an IR operation node.

    Args:
        node (IRNode): The IR operation node.
        target_framework (str): Target framework identifier.
        var_map (Optional[dict[str, str]]): Mapping from node IDs to variable names.

    Returns:
        list[cst.Arg]: Constructed positional and keyword arguments.
    """
    resolved_inputs: list[str] = [var_map.get(inp, inp) if var_map else inp for inp in node.inputs]
    args: list[cst.Arg] = [cst.Arg(value=cst.Name(inp)) for inp in resolved_inputs]
    kw_map: dict[str, str] = {}
    if target_framework in _CONFIG.frameworks:
        kw_map = _CONFIG.frameworks[target_framework].kwarg_map

    for k, v in node.attributes.items():
        if k in ("is_parameter", "param_name", "var_name", "dtype", "value", "shape", "callee", "has_varargs", "has_varkwargs", "is_ternary"):
            continue
        target_k = kw_map.get(k, k)
        if isinstance(v, bool):
            val_expr: cst.BaseExpression = cst.Name("True" if v else "False")
        elif isinstance(v, int):
            val_expr = cst.Integer(str(v))
        elif isinstance(v, float):
            val_expr = cst.Float(str(v))
        elif isinstance(v, str):
            val_expr = cst.SimpleString(f'"{v}"')
        else:
            name_str = str(v)
            val_expr = cst.Name(name_str if name_str.isidentifier() else "custom_val")
        args.append(cst.Arg(keyword=cst.Name(target_k), value=val_expr))

    return args


def _emit_control_flow_statement(
    node: IRNode,
    var_map: Optional[dict[str, str]] = None,
    assigned_name: Optional[str] = None,
) -> Optional[cst.SimpleStatementLine]:
    """Emit control flow statements for Cond, Scan, and WhileLoop nodes.

    Args:
        node (IRNode): The IR node.
        var_map (Optional[dict[str, str]]): Mapping from node IDs to variable names.
        assigned_name (Optional[str]): Target variable name for assignment.

    Returns:
        Optional[cst.SimpleStatementLine]: Emitted statement or None.
    """
    if node.op_type == "Cond":
        target_name = assigned_name or node.attributes.get("var_name", node.id)
        target = cst.AssignTarget(target=cst.Name(str(target_name)))
        if len(node.inputs) >= 3:
            cond_inp = var_map.get(node.inputs[0], node.inputs[0]) if var_map else node.inputs[0]
            true_inp = var_map.get(node.inputs[1], node.inputs[1]) if var_map else node.inputs[1]
            false_inp = var_map.get(node.inputs[2], node.inputs[2]) if var_map else node.inputs[2]
            cond_expr = cst.Name(cond_inp)
            true_expr = cst.Name(true_inp)
            false_expr = cst.Name(false_inp)
            if_exp = cst.IfExp(test=cond_expr, body=true_expr, orelse=false_expr)
            return cst.SimpleStatementLine(body=[cst.Assign(targets=[target], value=if_exp)])

    if node.op_type in ("Scan", "WhileLoop"):
        target_name = assigned_name or node.attributes.get("var_name", node.id)
        target = cst.AssignTarget(target=cst.Name(str(target_name)))
        raw_inp = node.inputs[0] if node.inputs else "items"
        iter_inp = var_map.get(raw_inp, raw_inp) if var_map else raw_inp
        scan_call = cst.Call(
            func=cst.Name("scan" if node.op_type == "Scan" else "while_loop"),
            args=[cst.Arg(value=cst.Name(iter_inp))],
        )
        return cst.SimpleStatementLine(body=[cst.Assign(targets=[target], value=scan_call)])

    return None


def _emit_container_statement(
    node: IRNode,
    var_map: Optional[dict[str, str]] = None,
    assigned_name: Optional[str] = None,
) -> Optional[cst.SimpleStatementLine]:
    """Emit container access statements for DictGet and GetItem nodes.

    Args:
        node (IRNode): The IR node.
        var_map (Optional[dict[str, str]]): Mapping from node IDs to variable names.
        assigned_name (Optional[str]): Target variable name for assignment.

    Returns:
        Optional[cst.SimpleStatementLine]: Emitted statement or None.
    """
    if node.op_type == "DictGet":
        target_name = assigned_name or node.attributes.get("var_name", node.id)
        target = cst.AssignTarget(target=cst.Name(str(target_name)))
        raw_inp = node.inputs[0] if node.inputs else "params"
        dict_name = var_map.get(raw_inp, raw_inp) if var_map else raw_inp
        key_str = str(node.attributes.get("key", "key"))
        subscript = cst.Subscript(
            value=cst.Name(dict_name),
            slice=[cst.SubscriptElement(slice=cst.Index(value=cst.SimpleString(f'"{key_str}"')))],
        )
        return cst.SimpleStatementLine(body=[cst.Assign(targets=[target], value=subscript)])

    if node.op_type == "GetItem":
        target_name = assigned_name or node.attributes.get("var_name", node.id)
        target = cst.AssignTarget(target=cst.Name(str(target_name)))
        raw_inp = node.inputs[0] if node.inputs else "arr"
        container_name = var_map.get(raw_inp, raw_inp) if var_map else raw_inp
        idx = node.attributes.get("index", 0)
        idx_expr = cst.Integer(str(idx)) if isinstance(idx, int) else cst.Name(str(idx))
        subscript = cst.Subscript(
            value=cst.Name(container_name),
            slice=[cst.SubscriptElement(slice=cst.Index(value=idx_expr))],
        )
        return cst.SimpleStatementLine(body=[cst.Assign(targets=[target], value=subscript)])

    return None


def _emit_foreign_statement(
    node: IRNode,
    var_map: Optional[dict[str, str]] = None,
    assigned_name: Optional[str] = None,
) -> Optional[cst.SimpleStatementLine]:
    """Emit ForeignCall statements forwarding varargs and kwargs.

    Args:
        node (IRNode): The IR node.
        var_map (Optional[dict[str, str]]): Mapping from node IDs to variable names.
        assigned_name (Optional[str]): Target variable name for assignment.

    Returns:
        Optional[cst.SimpleStatementLine]: Emitted statement or None.
    """
    if node.op_type != "ForeignCall":
        return None

    target_name = assigned_name if assigned_name is not None else node.attributes.get("var_name", node.id)
    callee = str(node.attributes.get("callee", "foreign_fn"))
    func_expr = _build_attribute_chain(callee.split("."))
    resolved_inputs = [var_map.get(inp, inp) if var_map else inp for inp in node.inputs]
    args: list[cst.Arg] = [cst.Arg(value=cst.Name(inp)) for inp in resolved_inputs]
    if node.attributes.get("has_varargs"):
        args.append(cst.Arg(value=cst.Name("args"), star="*"))
    if node.attributes.get("has_varkwargs"):
        args.append(cst.Arg(value=cst.Name("kwargs"), star="**"))
    call_expr = cst.Call(func=func_expr, args=args)
    if target_name:
        target = cst.AssignTarget(target=cst.Name(str(target_name)))
        return cst.SimpleStatementLine(body=[cst.Assign(targets=[target], value=call_expr)])
    return cst.SimpleStatementLine(body=[cst.Expr(value=call_expr)])


def _emit_operator_statement(
    node: IRNode,
    target_framework: str,
    var_map: Optional[dict[str, str]] = None,
    assigned_name: Optional[str] = None,
) -> Optional[cst.SimpleStatementLine]:
    """Emit mapped operator call statement.

    Args:
        node (IRNode): The IR node.
        target_framework (str): Target framework name.
        var_map (Optional[dict[str, str]]): Mapping from node IDs to variable names.
        assigned_name (Optional[str]): Target variable name for assignment.

    Returns:
        Optional[cst.SimpleStatementLine]: Emitted statement or None.
    """
    op_type: str = node.op_type
    if op_type in _CONFIG.ir_to_ast_ops and target_framework in _CONFIG.ir_to_ast_ops[op_type]:
        path: list[str] = _CONFIG.ir_to_ast_ops[op_type][target_framework]
        func_expr: cst.BaseExpression = _build_attribute_chain(path)
        call_args = _emit_call_args(node, target_framework, var_map=var_map)
        call: cst.Call = cst.Call(func=func_expr, args=call_args)

        target_name = assigned_name if assigned_name is not None else node.attributes.get("var_name")
        if target_name:
            target = cst.AssignTarget(target=cst.Name(str(target_name)))
            return cst.SimpleStatementLine(body=[cst.Assign(targets=[target], value=call)])
        return cst.SimpleStatementLine(body=[cst.Expr(value=call)])

    return None


def _emit_node_statement(
    node: IRNode,
    target_framework: str,
    var_map: Optional[dict[str, str]] = None,
    assigned_name: Optional[str] = None,
) -> Optional[cst.SimpleStatementLine]:
    """Emit CST statement for a single IR node.

    Args:
        node (IRNode): The IR node to convert.
        target_framework (str): Target framework identifier.
        var_map (Optional[dict[str, str]]): Mapping from node IDs to variable names.
        assigned_name (Optional[str]): Target variable name for assignment.

    Returns:
        Optional[cst.SimpleStatementLine]: Emitted statement or None.
    """
    if node.op_type == "Input":
        target_id = assigned_name or node.id
        assign_target = cst.AssignTarget(target=cst.Name(target_id))
        return cst.SimpleStatementLine(body=[cst.Assign(targets=[assign_target], value=cst.Name(target_id))])

    cf_stmt = _emit_control_flow_statement(node, var_map=var_map, assigned_name=assigned_name)
    if cf_stmt is not None:
        return cf_stmt

    cnt_stmt = _emit_container_statement(node, var_map=var_map, assigned_name=assigned_name)
    if cnt_stmt is not None:
        return cnt_stmt

    fn_stmt = _emit_foreign_statement(node, var_map=var_map, assigned_name=assigned_name)
    if fn_stmt is not None:
        return fn_stmt

    return _emit_operator_statement(node, target_framework, var_map=var_map, assigned_name=assigned_name)


def _build_var_map(graph: IRGraph) -> dict[str, str]:
    """Build a mapping from IR node IDs to target variable names.

    Args:
        graph (IRGraph): The computation graph.

    Returns:
        dict[str, str]: Mapping from node ID to variable name.
    """
    used_inputs: set[str] = set()
    for n in graph.nodes.values():
        for inp in n.inputs:
            used_inputs.add(inp)

    var_map: dict[str, str] = {}
    for nid, node in graph.nodes.items():
        if node.op_type == "Input":
            var_map[nid] = nid
        elif node.attributes.get("var_name"):
            var_map[nid] = str(node.attributes["var_name"])
        elif nid in used_inputs:
            var_map[nid] = nid
    return var_map


def emit_ir_to_ast(graph: IRGraph, target_framework: str) -> cst.Module:
    """Emit an IRGraph to a libcst AST Module with mapped arguments.

    Args:
        graph (IRGraph): The IR computation graph.
        target_framework (str): The target framework name (e.g., 'jax', 'pytorch').

    Returns:
        cst.Module: Resulting libcst AST module.
    """
    var_map: dict[str, str] = _build_var_map(graph)
    body: list[cst.SimpleStatementLine] = []

    for node_id, node in graph.nodes.items():
        assigned_name = var_map.get(node_id)
        stmt = _emit_node_statement(node, target_framework, var_map=var_map, assigned_name=assigned_name)
        if stmt is not None:
            body.append(stmt)

    return cst.Module(body=body)


def _get_class_base_expr(target_framework: str) -> cst.BaseExpression:
    """Resolve the target framework base class expression declaratively from config.

    Args:
        target_framework (str): Target framework identifier.

    Returns:
        cst.BaseExpression: CST expression for framework model base class.
    """
    fw_config = _CONFIG.frameworks.get(target_framework)
    if fw_config and fw_config.class_bases:
        base_parts = next(iter(fw_config.class_bases.values()), None)
        if base_parts:
            return _build_attribute_chain(base_parts)
    return cst.Name("object")


def _get_compute_method_name(target_framework: str) -> str:
    """Get the primary compute method name for the target framework declaratively from config.

    Args:
        target_framework (str): Target framework identifier.

    Returns:
        str: Method name.
    """
    defaults = {"pytorch": "forward", "jax": "__call__", "keras": "call"}
    fw_config = _CONFIG.frameworks.get(target_framework)
    if fw_config and fw_config.method_map:
        for key in ("__call__", "compute", "forward", "call"):
            if key in fw_config.method_map:
                return fw_config.method_map[key]
        return next(iter(fw_config.method_map.values()), defaults.get(target_framework, "forward"))
    return defaults.get(target_framework, "forward")


def _build_class_params(graph: IRGraph) -> list[cst.Param]:
    """Build the parameter list for the emitted class method.

    Args:
        graph (IRGraph): The computation graph.

    Returns:
        list[cst.Param]: Parameter list starting with self.
    """
    params: list[cst.Param] = [cst.Param(name=cst.Name("self"))]
    user_inputs = [n for n in graph.nodes.values() if n.op_type == "Input" and not n.attributes.get("is_parameter")]
    if user_inputs:
        for inp in user_inputs:
            params.append(cst.Param(name=cst.Name(inp.id)))
    else:
        params.append(cst.Param(name=cst.Name("x")))
    return params


def _build_return_statement(
    graph: IRGraph,
    var_map: dict[str, str],
    last_val_name: Optional[str],
) -> cst.SimpleStatementLine:
    """Build the final return or pass statement for a method.

    Args:
        graph (IRGraph): The computation graph.
        var_map (dict[str, str]): Variable name mapping.
        last_val_name (Optional[str]): Last evaluated variable name.

    Returns:
        cst.SimpleStatementLine: Return or Pass statement.
    """
    if graph.outputs:
        ret_target = graph.outputs[-1]
        out_node = graph.nodes.get(ret_target)
        ret_name = var_map.get(ret_target) or (out_node.attributes.get("var_name", ret_target) if out_node else ret_target)
        return cst.SimpleStatementLine(body=[cst.Return(value=cst.Name(str(ret_name)))])
    if last_val_name is not None:
        return cst.SimpleStatementLine(body=[cst.Return(value=cst.Name(last_val_name))])
    return cst.SimpleStatementLine(body=[cst.Pass()])


def emit_ir_to_class(
    graph: IRGraph,
    target_framework: str,
    class_name: str = "Model",
) -> cst.ClassDef:
    """Reconstruct target framework class structures from an IRGraph.

    Args:
        graph (IRGraph): The IR computation graph.
        target_framework (str): The target framework ('pytorch', 'jax', 'keras', etc.).
        class_name (str): The name for the emitted class definition.

    Returns:
        cst.ClassDef: Emitted libcst class definition node.
    """
    base_expr = _get_class_base_expr(target_framework)
    method_name = _get_compute_method_name(target_framework)
    params = _build_class_params(graph)
    var_map = _build_var_map(graph)

    for out_id in graph.outputs:
        if out_id in graph.nodes and out_id not in var_map:
            var_map[out_id] = out_id

    func_body: list[cst.BaseStatement] = []
    last_val_name: Optional[str] = None

    for node_id, node in graph.nodes.items():
        if node.op_type == "Input":
            continue
        assigned_name = var_map.get(node_id)
        stmt = _emit_node_statement(node, target_framework, var_map=var_map, assigned_name=assigned_name)
        if stmt is not None:
            func_body.append(stmt)
            last_val_name = str(assigned_name or node.attributes.get("var_name", node.id))

    func_body.append(_build_return_statement(graph, var_map, last_val_name))

    method_def = cst.FunctionDef(
        name=cst.Name(method_name),
        params=cst.Parameters(params=params),
        body=cst.IndentedBlock(body=func_body),
    )

    return cst.ClassDef(
        name=cst.Name(class_name),
        bases=[cst.Arg(value=base_expr)],
        body=cst.IndentedBlock(body=[method_def]),
    )
