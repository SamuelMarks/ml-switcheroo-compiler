"""IRGraph to AST emission."""

import os

import libcst as cst

from ml_switcheroo_compiler.backends.transpiler_config_models import load_transpiler_config
from ml_switcheroo_compiler.ir.core import IRGraph

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "transpilation_rules.yaml")
_CONFIG = load_transpiler_config(_CONFIG_PATH)


def _build_attribute_chain(names: list[str]) -> cst.BaseExpression:
    """Build the attribute chain."""
    if not names:
        return cst.Name("empty")
    if len(names) == 1:
        return cst.Name(names[0])
    expr: cst.BaseExpression = cst.Name(names[0])
    for name in names[1:]:
        expr = cst.Attribute(value=expr, attr=cst.Name(name))
    return expr


def emit_ir_to_ast(graph: IRGraph, target_framework: str) -> cst.Module:
    """Emit IRGraph to AST module.

    Args:
        graph (IRGraph): The IR graph.
        target_framework (str): The target framework.

    Returns:
        cst.Module: Resulting AST module.
    """
    body: list[cst.SimpleStatementLine] = []

    for _node_id, node in graph.nodes.items():
        op_type = getattr(node, "op_type", "")
        if op_type in _CONFIG.ir_to_ast_ops and target_framework in _CONFIG.ir_to_ast_ops[op_type]:
            path = _CONFIG.ir_to_ast_ops[op_type][target_framework]
            func_expr = _build_attribute_chain(path)
            call = cst.Call(func=func_expr, args=[])
            stmt = cst.SimpleStatementLine(body=[cst.Expr(value=call)])
            body.append(stmt)

    return cst.Module(body=body)
