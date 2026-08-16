"""AST to IRGraph parsing visitor."""

import os
import uuid

import libcst as cst

from ml_switcheroo_compiler.backends.transpiler_config_models import load_transpiler_config
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "transpilation_rules.yaml")
_CONFIG = load_transpiler_config(_CONFIG_PATH)


class ASTToIRVisitor(cst.CSTVisitor):
    """Parses framework-specific AST nodes into the Unified IRGraph."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.graph = IRGraph()
        self.current_id = 0

    def _get_base_name(self, node: cst.BaseExpression) -> str:
        """Get the base name of a node."""
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            return f"{self._get_base_name(node.value)}.{node.attr.value}"
        return ""

    def visit_Call(self, node: cst.Call) -> None:
        """Visit call node."""
        if isinstance(node.func, (cst.Name, cst.Attribute)):
            full_name = self._get_base_name(node.func)
            if full_name in _CONFIG.ast_to_ir_ops:
                op_type = _CONFIG.ast_to_ir_ops[full_name]
                node_id = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
                self.current_id += 1
                ir_node = IRNode(id=node_id, op_type=op_type, inputs=[])
                self.graph.nodes[node_id] = ir_node


def parse_ast_to_ir(source_code: str) -> IRGraph:
    """Parse source code to IRGraph.

    Args:
        source_code (str): Source code.

    Returns:
        IRGraph: Resulting graph.
    """
    tree = cst.parse_module(source_code)
    visitor = ASTToIRVisitor()
    tree.visit(visitor)
    return visitor.graph
