"""AST to IRGraph parsing visitor."""

import os
import uuid
from collections.abc import Sequence
from typing import Optional

import libcst as cst

from ml_switcheroo_compiler.backends.transpiler_config_models import load_transpiler_config
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "transpilation_rules.yaml")
_CONFIG = load_transpiler_config(_CONFIG_PATH)


class ASTToIRVisitor(cst.CSTVisitor):
    """Parses framework-specific AST nodes into the Unified IRGraph."""

    def __init__(self, parent_scope: Optional[dict[str, str]] = None) -> None:
        """Initialize the visitor.

        Args:
            parent_scope (Optional[dict[str, str]]): Outer scope variable bindings for closure analysis.
        """
        super().__init__()
        self.graph = IRGraph()
        self.current_id = 0
        self.var_table: dict[str, str] = {}
        self.last_node_id: Optional[str] = None
        self.parent_scope: Optional[dict[str, str]] = parent_scope

    def _lookup_var(self, name: str) -> str:
        """Lookup variable name, capturing from parent closure scope if necessary.

        Args:
            name (str): The variable name.

        Returns:
            str: Resolved node ID.
        """
        if name in self.var_table:
            return self.var_table[name]
        if self.parent_scope and name in self.parent_scope:
            if name not in self.graph.nodes:
                self.graph.nodes[name] = IRNode(id=name, op_type="Input", inputs=[])
                if hasattr(self.graph, "inputs") and isinstance(self.graph.inputs, list):
                    if name not in self.graph.inputs:
                        self.graph.inputs.append(name)
            self.var_table[name] = name
            return name
        return name

    def _get_base_name(self, node: cst.BaseExpression) -> str:
        """Get the dot-separated qualified name of an expression.

        Args:
            node (cst.BaseExpression): CST expression.

        Returns:
            str: Qualified attribute chain string.
        """
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            base = self._get_base_name(node.value)
            return f"{base}.{node.attr.value}" if base else node.attr.value
        return ""

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Register function parameters as explicit IR input nodes.

        Args:
            node (cst.FunctionDef): Function definition AST node.
        """
        for param in node.params.params:
            param_name = param.name.value
            if param_name != "self" and param_name not in self.graph.nodes:
                self.graph.nodes[param_name] = IRNode(id=param_name, op_type="Input", inputs=[])
                self.var_table[param_name] = param_name

    def _handle_self_assign(self, target: cst.Attribute, node: cst.Assign) -> bool:
        """Handle assignment to self.<attr> representing class parameters.

        Args:
            target (cst.Attribute): Attribute target.
            node (cst.Assign): Parent assign statement.

        Returns:
            bool: True if handled as self parameter.
        """
        if isinstance(target.value, cst.Name) and target.value.value == "self":
            attr_name = target.attr.value
            val: Optional[object] = None
            if isinstance(node.value, cst.Integer):
                val = int(node.value.value)
            elif isinstance(node.value, cst.Float):
                val = float(node.value.value)

            param_id = f"param_{attr_name}"
            ir_node = IRNode(
                id=param_id,
                op_type="Input",
                inputs=[],
                attributes={"is_parameter": True, "param_name": attr_name, "value": val},
            )
            self.graph.nodes[param_id] = ir_node
            self.var_table[f"self.{attr_name}"] = param_id
            return True
        return False

    def _recursive_unpack(self, target_container: cst.BaseExpression, val_node_id: str, prefix: str = "") -> None:
        """Recursively unpack nested tuples or lists into GetItem IR nodes.

        Args:
            target_container (cst.BaseExpression): Container target expression.
            val_node_id (str): Parent value node ID.
            prefix (str): Index path prefix.
        """
        if not isinstance(target_container, (cst.Tuple, cst.List)):
            return
        for i, element in enumerate(target_container.elements):
            idx_str = f"{prefix}_{i}" if prefix else str(i)
            item_node_id = f"{val_node_id}_item_{self.current_id}_{idx_str}"
            self.current_id += 1
            if isinstance(element.value, cst.Name):
                elem_name = element.value.value
                item_node = IRNode(
                    id=item_node_id,
                    op_type="GetItem",
                    inputs=[val_node_id],
                    attributes={"index": i, "var_name": elem_name},
                )
                self.graph.nodes[item_node_id] = item_node
                self.var_table[elem_name] = item_node_id
            elif isinstance(element.value, (cst.Tuple, cst.List)):
                item_node = IRNode(
                    id=item_node_id,
                    op_type="GetItem",
                    inputs=[val_node_id],
                    attributes={"index": i},
                )
                self.graph.nodes[item_node_id] = item_node
                self._recursive_unpack(element.value, item_node_id, idx_str)

    def _resolve_target_val_node_id(self, node: cst.Assign) -> Optional[str]:
        """Resolve node ID of the RHS value expression in an assignment.

        Args:
            node (cst.Assign): Parent assign statement.

        Returns:
            Optional[str]: Node ID if resolvable.
        """
        if isinstance(node.value, cst.Name):
            return self._lookup_var(node.value.value)
        if isinstance(node.value, cst.Attribute):
            return self._lookup_var(self._get_base_name(node.value))
        self.last_node_id = None
        node.value.visit(self)
        return self.last_node_id

    def _handle_dict_unpack(self, target_expr: cst.Dict, val_node_id: str) -> None:
        """Handle dictionary unpacking assignment.

        Args:
            target_expr (cst.Dict): Dict target expression.
            val_node_id (str): Value node ID.
        """
        for i, element in enumerate(target_expr.elements):
            if isinstance(element, cst.DictElement) and isinstance(element.value, cst.Name):
                key_repr = element.key.value if isinstance(element.key, cst.SimpleString) else str(i)
                elem_name = element.value.value
                item_node_id = f"{val_node_id}_dict_{key_repr}"
                item_node = IRNode(
                    id=item_node_id,
                    op_type="GetItem",
                    inputs=[val_node_id],
                    attributes={"key": key_repr, "var_name": elem_name},
                )
                self.graph.nodes[item_node_id] = item_node
                self.var_table[elem_name] = item_node_id

    def _handle_unpack_assign(self, target_expr: cst.BaseAssignTargetExpression, node: cst.Assign) -> bool:
        """Handle container unpacking assignment into GetItem nodes.

        Args:
            target_expr (cst.BaseAssignTargetExpression): Unpack target (Tuple, List, or Dict).
            node (cst.Assign): Parent assign statement.

        Returns:
            bool: True if handled as unpack.
        """
        if isinstance(target_expr, (cst.Tuple, cst.List)):
            val_node_id = self._resolve_target_val_node_id(node)
            if val_node_id:
                self._recursive_unpack(target_expr, val_node_id)
            return True
        if isinstance(target_expr, cst.Dict):
            val_node_id = self._resolve_target_val_node_id(node)
            if val_node_id:
                self._handle_dict_unpack(target_expr, val_node_id)
            return True
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:
        """Visit assignment to track variable references and container unpacks.

        Args:
            node (cst.Assign): The assignment node.

        Returns:
            bool: False to stop recursive AST child visiting.
        """
        first_target = node.targets[0].target
        if isinstance(first_target, cst.Attribute) and self._handle_self_assign(first_target, node):
            return False
        if self._handle_unpack_assign(first_target, node):
            return False

        self.last_node_id = None
        node.value.visit(self)
        if self.last_node_id:
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    var_name = target.target.value
                    self.var_table[var_name] = self.last_node_id
                    self.graph.nodes[self.last_node_id].attributes["var_name"] = var_name

        return False

    def visit_Return(self, node: cst.Return) -> bool:
        """Visit return statement to capture graph outputs.

        Args:
            node (cst.Return): Return statement node.

        Returns:
            bool: False to halt duplicate child traversal in visitor.
        """
        self.last_node_id = None
        if isinstance(node.value, cst.Name) and node.value.value in self.var_table:
            ret_id = self.var_table[node.value.value]
            self.graph.outputs.append(ret_id)
        elif node.value:
            node.value.visit(self)
            if self.last_node_id:
                self.graph.outputs.append(self.last_node_id)
        return False

    def _resolve_call_arg(self, arg_val: cst.BaseExpression, inputs: list[str]) -> None:
        """Resolve a single positional call argument into inputs list.

        Args:
            arg_val (cst.BaseExpression): The argument expression.
            inputs (list[str]): Output list to append input node id.
        """
        if isinstance(arg_val, cst.Name):
            inputs.append(self._lookup_var(arg_val.value))
        elif isinstance(arg_val, cst.Attribute):
            attr_name = self._get_base_name(arg_val)
            inputs.append(self._lookup_var(attr_name))
        elif isinstance(arg_val, cst.Call):
            self.last_node_id = None
            arg_val.visit(self)
            if self.last_node_id:
                inputs.append(self.last_node_id)
        elif isinstance(arg_val, (cst.Integer, cst.Float)):
            val = int(arg_val.value) if isinstance(arg_val, cst.Integer) else float(arg_val.value)
            const_id = f"const_{self.current_id}_{uuid.uuid4().hex[:6]}"
            self.current_id += 1
            ir_node = IRNode(
                id=const_id,
                op_type="Constant",
                inputs=[],
                attributes={"value": val, "dtype": "float32" if isinstance(val, float) else "int64"},
            )
            self.graph.nodes[const_id] = ir_node
            inputs.append(const_id)
        else:
            inputs.append("unknown")

    def _parse_call_args(
        self,
        args: Sequence[cst.Arg],
        inputs: list[str],
        attributes: dict[str, object],
    ) -> None:
        """Parse positional, keyword, and varargs call arguments.

        Args:
            args (Sequence[cst.Arg]): Call arguments sequence.
            inputs (list[str]): Inputs list to populate.
            attributes (dict[str, object]): Attributes dictionary to populate.
        """
        for arg in args:
            if arg.star == "*":
                attributes["has_varargs"] = True
                self._resolve_call_arg(arg.value, inputs)
            elif arg.star == "**":
                attributes["has_varkwargs"] = True
                self._resolve_call_arg(arg.value, inputs)
            elif arg.keyword:
                kw = arg.keyword.value
                val: Optional[object] = None
                if isinstance(arg.value, cst.Integer):
                    val = int(arg.value.value)
                elif isinstance(arg.value, cst.Float):
                    val = float(arg.value.value)
                elif isinstance(arg.value, cst.SimpleString):
                    val = arg.value.evaluated_value
                elif isinstance(arg.value, cst.Name):
                    val = arg.value.value
                attributes[kw] = val
            else:
                self._resolve_call_arg(arg.value, inputs)

    def visit_Call(self, node: cst.Call) -> bool:
        """Visit call node to reconstruct IR operations.

        Args:
            node (cst.Call): Call AST node.

        Returns:
            bool: False to stop default child traversal.
        """
        if not isinstance(node.func, (cst.Name, cst.Attribute)):
            return False

        full_name: str = self._get_base_name(node.func)
        has_star: bool = any(arg.star in ("*", "**") for arg in node.args)
        if full_name not in _CONFIG.ast_to_ir_ops and not has_star:
            return False

        op_type: str = _CONFIG.ast_to_ir_ops.get(full_name, "ForeignCall")
        node_id: str = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        inputs: list[str] = []
        attributes: dict[str, object] = {}
        if op_type == "ForeignCall":
            attributes["callee"] = full_name

        self._parse_call_args(node.args, inputs, attributes)

        ir_node: IRNode = IRNode(id=node_id, op_type=op_type, inputs=inputs, attributes=attributes)
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id
        return False

    def visit_IfExp(self, node: cst.IfExp) -> None:
        """Visit Python ternary conditional expression to emit Cond IR node.

        Args:
            node (cst.IfExp): The ternary conditional AST node.
        """
        node_id: str = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        inputs: list[str] = []
        if isinstance(node.test, cst.Name) and node.test.value in self.var_table:
            inputs.append(self.var_table[node.test.value])
        else:
            self.last_node_id = None
            node.test.visit(self)
            if self.last_node_id:
                inputs.append(self.last_node_id)

        if isinstance(node.body, cst.Name) and node.body.value in self.var_table:
            inputs.append(self.var_table[node.body.value])
        else:
            self.last_node_id = None
            node.body.visit(self)
            if self.last_node_id:
                inputs.append(self.last_node_id)

        if isinstance(node.orelse, cst.Name) and node.orelse.value in self.var_table:
            inputs.append(self.var_table[node.orelse.value])
        else:
            self.last_node_id = None
            node.orelse.visit(self)
            if self.last_node_id:
                inputs.append(self.last_node_id)

        ir_node = IRNode(id=node_id, op_type="Cond", inputs=inputs, attributes={"is_ternary": True})
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_ListComp(self, node: cst.ListComp) -> None:
        """Visit Python list comprehension to emit Map / Scan IR nodes.

        Args:
            node (cst.ListComp): The list comprehension AST node.
        """
        node_id: str = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        iter_name = ""
        if isinstance(node.for_in.iter, cst.Name) and node.for_in.iter.value in self.var_table:
            iter_name = self.var_table[node.for_in.iter.value]
        else:
            self.last_node_id = None
            node.for_in.iter.visit(self)
            if self.last_node_id:
                iter_name = self.last_node_id

        inputs = [iter_name] if iter_name else []
        ir_node = IRNode(
            id=node_id,
            op_type="Map",
            inputs=inputs,
            attributes={"target_var": getattr(node.for_in.target, "value", "x")},
        )
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_If(self, node: cst.If) -> None:
        """Visit Python if statement to emit Cond IR nodes with recursive branch subgraphs.

        Args:
            node (cst.If): The If statement AST node.
        """
        node_id: str = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        inputs: list[str] = []
        if isinstance(node.test, cst.Name) and node.test.value in self.var_table:
            inputs.append(self.var_table[node.test.value])
        else:
            self.last_node_id = None
            node.test.visit(self)
            if self.last_node_id:
                inputs.append(self.last_node_id)

        true_visitor = ASTToIRVisitor()
        true_visitor.var_table = dict(self.var_table)
        node.body.visit(true_visitor)
        true_branch = true_visitor.graph

        false_branch: Optional[IRGraph] = None
        if node.orelse is not None:
            false_visitor = ASTToIRVisitor()
            false_visitor.var_table = dict(self.var_table)
            node.orelse.visit(false_visitor)
            false_branch = false_visitor.graph

        for k, v in true_visitor.var_table.items():
            if k not in self.var_table:
                self.var_table[k] = v

        attributes: dict[str, object] = {
            "true_branch": true_branch,
            "false_branch": false_branch,
        }
        ir_node: IRNode = IRNode(id=node_id, op_type="Cond", inputs=inputs, attributes=attributes)
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_While(self, node: cst.While) -> None:
        """Visit Python while statement to emit WhileLoop IR nodes with body subgraphs.

        Args:
            node (cst.While): The While statement AST node.
        """
        node_id: str = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        inputs: list[str] = []
        if isinstance(node.test, cst.Name) and node.test.value in self.var_table:
            inputs.append(self.var_table[node.test.value])
        else:
            self.last_node_id = None
            node.test.visit(self)
            if self.last_node_id:
                inputs.append(self.last_node_id)

        body_visitor = ASTToIRVisitor()
        body_visitor.var_table = dict(self.var_table)
        node.body.visit(body_visitor)
        body_subgraph = body_visitor.graph

        for k, v in body_visitor.var_table.items():
            if k not in self.var_table:
                self.var_table[k] = v

        attributes: dict[str, object] = {
            "body_subgraph": body_subgraph,
        }
        ir_node: IRNode = IRNode(id=node_id, op_type="WhileLoop", inputs=inputs, attributes=attributes)
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_For(self, node: cst.For) -> None:
        """Visit Python for loop statement to emit Scan or Loop IR nodes with body subgraphs.

        Args:
            node (cst.For): The For statement AST node.
        """
        node_id: str = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        inputs: list[str] = []
        if isinstance(node.iter, cst.Name) and node.iter.value in self.var_table:
            inputs.append(self.var_table[node.iter.value])
        else:
            self.last_node_id = None
            node.iter.visit(self)
            if self.last_node_id:
                inputs.append(self.last_node_id)

        target_name = node.target.value if isinstance(node.target, cst.Name) else "elem"
        body_visitor = ASTToIRVisitor()
        body_visitor.var_table = dict(self.var_table)
        body_visitor.var_table[target_name] = f"{node_id}_iter_val"
        node.body.visit(body_visitor)
        body_subgraph = body_visitor.graph

        attributes: dict[str, object] = {
            "body_subgraph": body_subgraph,
            "target_var": target_name,
        }
        ir_node: IRNode = IRNode(id=node_id, op_type="Scan", inputs=inputs, attributes=attributes)
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_Try(self, node: cst.Try) -> None:
        """Visit Python try/except statements to emit TryExcept IR nodes.

        Args:
            node (cst.Try): The Try statement AST node.
        """
        node_id: str = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        try_visitor = ASTToIRVisitor(parent_scope=dict(self.var_table))
        node.body.visit(try_visitor)
        try_subgraph = try_visitor.graph

        fallback_subgraph: Optional[IRGraph] = None
        if node.handlers:
            handler_visitor = ASTToIRVisitor(parent_scope=dict(self.var_table))
            node.handlers[0].body.visit(handler_visitor)
            fallback_subgraph = handler_visitor.graph

        for k, v in try_visitor.var_table.items():
            if k not in self.var_table:
                self.var_table[k] = v

        attributes: dict[str, object] = {
            "try_subgraph": try_subgraph,
            "fallback_subgraph": fallback_subgraph,
        }
        ir_node: IRNode = IRNode(id=node_id, op_type="TryExcept", inputs=[], attributes=attributes)
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_Raise(self, node: cst.Raise) -> None:
        """Visit Python raise statements to emit Assert or Raise IR nodes.

        Args:
            node (cst.Raise): The Raise statement AST node.
        """
        node_id: str = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        exc_name = "Exception"
        if node.exc and isinstance(node.exc, cst.Call) and isinstance(node.exc.func, cst.Name):
            exc_name = node.exc.func.value
        elif node.exc and isinstance(node.exc, cst.Name):
            exc_name = node.exc.value

        ir_node: IRNode = IRNode(
            id=node_id,
            op_type="Assert",
            inputs=[],
            attributes={"exception": exc_name},
        )
        self.graph.nodes[node_id] = ir_node
        self.last_node_id = node_id

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> bool:
        """Visit binary operations."""
        op_map = {
            cst.Add: "Add",
            cst.Subtract: "Sub",
            cst.Multiply: "Mul",
            cst.Divide: "Div",
            cst.Power: "Pow",
            cst.Modulo: "Mod",
        }
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
        """Visit slicing, dictionary indexing, and item access.

        Args:
            node (cst.Subscript): The Subscript AST node.
        """
        node_id = f"node_{self.current_id}_{uuid.uuid4().hex[:6]}"
        self.current_id += 1

        inputs = []
        if isinstance(node.value, cst.Name) and node.value.value in self.var_table:
            inputs.append(self.var_table[node.value.value])

        op_type = "Slice"
        attributes: dict[str, object] = {}

        if node.slice and len(node.slice) == 1:
            slice_elem = node.slice[0].slice
            if isinstance(slice_elem, cst.Index):
                if isinstance(slice_elem.value, cst.SimpleString):
                    op_type = "DictGet"
                    attributes["key"] = slice_elem.value.evaluated_value
                elif isinstance(slice_elem.value, cst.Integer):
                    op_type = "GetItem"
                    attributes["index"] = int(slice_elem.value.value)
                elif isinstance(slice_elem.value, cst.Name):
                    op_type = "GetItem"
                    if slice_elem.value.value in self.var_table:
                        inputs.append(self.var_table[slice_elem.value.value])
                    attributes["index_var"] = slice_elem.value.value

        ir_node: IRNode = IRNode(id=node_id, op_type=op_type, inputs=inputs, attributes=attributes)
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
    if not visitor.graph.outputs and visitor.last_node_id:
        visitor.graph.outputs.append(visitor.last_node_id)
    return visitor.graph
