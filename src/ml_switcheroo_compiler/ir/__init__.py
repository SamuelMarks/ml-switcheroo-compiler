"""Compiler-local Tier-2 IR utilities, symbolic shape algebra, and state node factories.

Shared Tier-1 Unified IR models (``LogicalGraph``, ``LogicalNode``, ``IRGraph``,
``IRNode``, ``TensorShape``, ``TensorSpec``, ``PartitionSpec``, ``AttributeValue``,
``ZeroTangent``, ``NoTangent``, ``NodeDict``, ``topological_sort``) live in the
external ``ml_switcheroo_ir`` package and must be imported directly from there.
"""

from __future__ import annotations

from ml_switcheroo_compiler.ir.callback_registry import (
    CustomRuleHandle,
    clear_custom_rule_handles,
    register_custom_rule_handle,
    resolve_custom_rule_handle,
)
from ml_switcheroo_compiler.ir.core import (
    AxesLike,
    IRAttrValue,
    ScalarLike,
    ShapeLike,
    SymbolicDim,
    as_node_list,
    as_opt_graph,
    as_opt_pspec,
    clone_logical_node,
    ensure_attr_val,
    ensure_attrs,
    ensure_graph,
    normalize_shape_metadata,
)
from ml_switcheroo_compiler.ir.shape_system import (
    AlgebraicRuleModel,
    AlgebraicRuleSetModel,
    Polynomial,
    ShapeTracker,
    SymBinaryOp,
    SymbolicConstraintTracker,
    SymConst,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
    load_algebraic_rules,
)
from ml_switcheroo_compiler.ir.state import (
    ASSIGN_OP,
    ASSIGN_VARIABLE_OP,
    READ_VARIABLE_OP,
    SCATTER_UPDATE_OP,
    create_assign_variable,
    create_read_variable,
    create_scatter_update,
)

__all__ = [
    "ASSIGN_OP",
    "ASSIGN_VARIABLE_OP",
    "AlgebraicRuleModel",
    "AlgebraicRuleSetModel",
    "AxesLike",
    "CustomRuleHandle",
    "IRAttrValue",
    "Polynomial",
    "READ_VARIABLE_OP",
    "SCATTER_UPDATE_OP",
    "ScalarLike",
    "ShapeLike",
    "ShapeTracker",
    "SymBinaryOp",
    "SymConst",
    "SymInt",
    "SymNode",
    "SymPiecewise",
    "SymUnaryOp",
    "SymVar",
    "SymbolicConstraintTracker",
    "SymbolicDim",
    "as_node_list",
    "as_opt_graph",
    "as_opt_pspec",
    "clear_custom_rule_handles",
    "clone_logical_node",
    "create_assign_variable",
    "create_read_variable",
    "create_scatter_update",
    "ensure_attr_val",
    "ensure_attrs",
    "ensure_graph",
    "load_algebraic_rules",
    "normalize_shape_metadata",
    "register_custom_rule_handle",
    "resolve_custom_rule_handle",
]
