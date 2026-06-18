"""Code generator visitor for traversing IR."""

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.backends.backend_utils import resolve_input_vars, format_shape_metadata

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator


class CodeGeneratorVisitor:
    """Base visitor for code generation."""

    def __init__(self, generator: "BaseGenerator") -> None:
        """Initializes the object.

        Args:
            generator (Any): The parent generator object.
        """
        self.generator = generator

    def generate_body(self, input_prefix: str = "args") -> None:
        """Visit nodes to generate code body.

        Args:
            input_prefix (str): The input_prefix parameter for the operation.
        """
        self.generator.input_idx = 0
        self.generator._output_returns = []

        for node in self.generator.sorted_nodes:
            self.generate_node(node, input_prefix)

        self.generator._emit_body_return(self.generator._output_returns)

    def generate_node(self, node: IRNode, input_prefix: str) -> None:
        """Execute _generate_node.

        Args:
            node (Any): Argument node.
            input_prefix (Any): Argument input_prefix.
        """
        if node.op_type == "Constant":
            self.handle_constant_node(node)
        elif node.op_type == "Input":
            self.handle_input_node(node, input_prefix)
        elif node.op_type == "Output":
            self.handle_output_node(node)
        else:
            self.handle_compute_node(node)

    def handle_constant_node(self, node: IRNode) -> None:
        """Execute _handle_constant_node.

        Args:
            node (Any): Argument node.
        """
        val_repr = self.generator.emit_constant(node)
        var_name = self.generator.assign_var_name(node.id, "const")
        self.generator._emit_constant_assignment(var_name, val_repr)

    def handle_input_node(self, node: IRNode, input_prefix: str) -> None:
        """Execute _handle_input_node.

        Args:
            node (Any): Argument node.
            input_prefix (Any): Argument input_prefix.
        """
        var_name = self.generator.assign_var_name(node.id, "input")
        self.generator._emit_input_assignment(
            var_name, node, input_prefix, self.generator.input_idx
        )
        self.generator.input_idx += 1

    def handle_output_node(self, node: IRNode) -> None:
        """Execute _handle_output_node.

        Args:
            node (Any): Argument node.
        """
        input_vars = resolve_input_vars(node, self.generator.var_names)
        returns = ", ".join(input_vars)
        self.generator._emit_output_assignment(node, input_vars, returns)

    def handle_compute_node(self, node: IRNode) -> None:
        """Execute _handle_compute_node.

        Args:
            node (Any): Argument node.
        """
        var_name = self.generator.assign_var_name(node.id)
        input_vars = resolve_input_vars(node, self.generator.var_names)

        kwargs = {**node.attributes}
        if "stream_id" in node.attributes:
            kwargs["stream_id"] = node.attributes["stream_id"]
        if "async_check" in node.attributes:
            kwargs["async_check"] = node.attributes["async_check"]

        shape_str = format_shape_metadata(node, self.generator.var_names)
        if shape_str is not None:
            kwargs["shape"] = shape_str

        expr = self.generator.visit(node, input_vars, **kwargs)
        self.generator.add_line(f"{var_name} = {expr}")
