"""AST to IRGraph parsing visitor."""

import os
import uuid
from typing import Optional, cast

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
        self.var_table: dict[str, str] = {}
        self.last_node_id: Optional[str] = None

    def _get_base_name(self, node: cst.BaseExpression) -> str:
        """Get the base name of a node."""
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            return f"{self._get_base_name(cast(cst.BaseExpression, node.value))}.{node.attr.value}"
        return ""

    def visit_Assign(self, node: cst.Assign) -> None:
        """Visit assignment to track variable references."""
        self.last_node_id = None
        node.value.visit(self)
        if self.last_node_id:
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    self.var_table[target.target.value] = self.last_node_id

        # Prevent default children visiting
        return False

    def visit_Call(self, node: cst.Call) -> None:
        """Visit call node."""
        if isinstance(node.func, (cst.Name, cst.Attribute)):
            full_name: str = self._get_base_name(node.func)
            if full_name in _CONFIG.ast_to_ir_ops:
                op_type: str = _CONFIG.ast_to_ir_ops[full_name]
                node_id: str = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
                self.current_id += 1

                # Resolve arguments to establish edges
                inputs = []
                for arg in node.args:
                    if isinstance(arg.value, cst.Name):
                        inputs.append(self.var_table.get(arg.value.value, "unknown"))
                    elif isinstance(arg.value, cst.Call):
                        # Recursive evaluation for inline calls
                        self.last_node_id = None
                        arg.value.visit(self)
                        if self.last_node_id:
                            inputs.append(self.last_node_id)

                ir_node: IRNode = IRNode(id=node_id, op_type=op_type, inputs=inputs)
                self.graph.nodes[node_id] = ir_node
                self.last_node_id = node_id

        return False

    def visit_If(self, node: cst.If) -> None:
        """Visit Python if statement to emit Cond IR nodes."""
        node_id = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        # Simple conditional capture
        inputs = []
        if isinstance(node.test, cst.Name) and node.test.value in self.var_table:
            inputs.append(self.var_table[node.test.value])

        ir_node: IRNode = IRNode(id=node_id, op_type="Cond", inputs=inputs)
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_While(self, node: cst.While) -> None:
        """Visit Python while statement to emit WhileLoop IR nodes."""
        node_id = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        inputs = []
        if isinstance(node.test, cst.Name) and node.test.value in self.var_table:
            inputs.append(self.var_table[node.test.value])

        ir_node: IRNode = IRNode(id=node_id, op_type="WhileLoop", inputs=inputs)
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
        """Visit binary operations."""
        op_map = {cst.Add: "Add", cst.Subtract: "Sub", cst.Multiply: "Mul", cst.Divide: "Div", cst.Power: "Pow", cst.Modulo: "Mod"}
        op_type = "UnknownBinaryOp"
        for cst_type, ir_type in op_map.items():
            if isinstance(node.operator, cst_type):
                op_type = ir_type
                break

        node_id = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        inputs = []
        for child in (node.left, node.right):
            if isinstance(child, cst.Name) and child.value in self.var_table:
                inputs.append(self.var_table[child.value])
            else:
                self.last_node_id = None
                child.visit(self)
                if self.last_node_id:
                    inputs.append(self.last_node_id)

        ir_node: IRNode = IRNode(id=node_id, op_type=op_type, inputs=inputs)
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

        return False

    def visit_Integer(self, node: cst.Integer) -> None:
        """Visit Integer literal."""
        node_id = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        ir_node: IRNode = IRNode(id=node_id, op_type="Constant", inputs=[], attributes={"value": int(node.value), "dtype": "int64"})
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_Float(self, node: cst.Float) -> None:
        """Visit Float literal."""
        node_id = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        ir_node: IRNode = IRNode(id=node_id, op_type="Constant", inputs=[], attributes={"value": float(node.value), "dtype": "float32"})
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_Subscript(self, node: cst.Subscript) -> None:
        """Visit slicing/indexing."""
        node_id = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        inputs = []
        if isinstance(node.value, cst.Name) and node.value.value in self.var_table:
            inputs.append(self.var_table[node.value.value])

        ir_node: IRNode = IRNode(id=node_id, op_type="Slice", inputs=inputs)
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id


def parse_ast_to_ir(source_code: str) -> IRGraph:
    """Parse source code to IRGraph.

    Args:
        source_code (str): Source code.

    Returns:
        IRGraph: Resulting graph.
    """
    tree: cst.Module = cst.parse_module(source_code)
    visitor: ASTToIRVisitor = ASTToIRVisitor()
    tree.visit(visitor)
    return visitor.graph
