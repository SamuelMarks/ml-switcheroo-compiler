"""Base generator for emitting backend code from IR."""

from abc import ABC, abstractmethod
from typing import Union

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

from ml_switcheroo_compiler.backends.formatters import CodeFormatter
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor


class BaseGenerator(ABC):
    """Abstract base class for backend code generation."""

    def __init__(self, graph: IRGraph) -> None:
        """Initializes the object.

        Args:
            graph (IRGraph): The graph to process.
        """
        self.graph = graph

        self.sorted_nodes = topological_sort(graph)
        self.formatter = CodeFormatter()

    @property
    def var_names(self) -> dict[str, str]:
        """Proxy property for formatter var_names."""
        return self.formatter.var_names

    @var_names.setter
    def var_names(self, value: dict[str, str]) -> None:
        """Proxy property for formatter var_names."""
        self.formatter.var_names = value

    @property
    def code(self) -> list[str]:
        """Proxy property for formatter code."""
        return self.formatter.code

    @code.setter
    def code(self, value: list[str]) -> None:
        """Proxy property for formatter code."""
        self.formatter.code = value

    @property
    def indent_level(self) -> int:
        """Proxy property for formatter indent_level."""
        return self.formatter.indent_level

    @indent_level.setter
    def indent_level(self, value: int) -> None:
        """Proxy property for formatter indent_level."""
        self.formatter.indent_level = value

    @property
    def header(self) -> str:
        """Proxy property for formatter header."""
        return self.formatter.header

    @header.setter
    def header(self, value: str) -> None:
        """Proxy property for formatter header."""
        self.formatter.header = value

    def get_indent(self) -> str:
        """Evaluate get indent."""
        return self.formatter.get_indent()

    def add_line(self, line: str) -> None:
        """Add line."""
        self.formatter.add_line(line)

    def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
        """Assign var name."""
        return self.formatter.assign_var_name(node_id, prefix)

    def emit_constant(self, node: IRNode) -> str:
        """Emit code for the constant backend.

        Args:
            node (IRNode): The node to process.

        Returns:
            str: The evaluated output resulting from this operation.
        """
        val = node.attributes.get("value")
        return repr(val)

    @abstractmethod
    def generate(self) -> str:
        """Define the function wrapper.

        Returns:
            str: The evaluated output resulting from this operation.
        """
        ...

    def _generate_body(self, input_prefix: str = "args") -> None:
        """Visit nodes to generate code body.

        Args:
            input_prefix (str): The input_prefix parameter for the operation.
        """
        visitor = CodeGeneratorVisitor(self)
        visitor.generate_body(input_prefix)

    def _emit_body_return(self, returns: list[str]) -> None:
        """Emit the final return statement.

        Args:
            returns: List of return variable names.
        """
        if returns:
            if len(returns) == 1:
                self.add_line(f"return {returns[0]}")
            else:
                self.add_line(f"return ({', '.join(returns)})")
        else:
            self.add_line("return None")

    def visit(
        self,
        node: IRNode,
        input_vars: list[str],
        **kwargs: object,
    ) -> str:
        """Visit a node and return the formatted code string for the operation.

        Args:
            node (IRNode): The node parameter for the operation.
            input_vars (list[str]): The input_vars parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The evaluated output resulting from this operation.
        """
        op_type = getattr(node, "op_type", "")
        method_name = f"visit_{op_type}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(node, input_vars, **kwargs)
        return self.generic_visit(node, input_vars, **kwargs)

    @abstractmethod
    def generic_visit(
        self,
        node: IRNode,
        input_vars: list[str],
        **kwargs: object,
    ) -> str:
        """Fallback visit method for operations not explicitly handled.

        Args:
            node (IRNode): The node.
            input_vars (list[str]): The inputs.
            **kwargs: Additional attributes.

        Returns:
            str: The code string.
        """
        ...

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Override in subclasses to emit framework-specific constant arrays.

        Args:
            var_name (str): The var_name parameter for the operation.
            val_repr (str): The val_repr parameter for the operation.
        """
        self.add_line(f"{var_name} = {val_repr}")

    def _emit_input_assignment(
        self,
        var_name: str,
        node: IRNode,
        input_prefix: str,
        input_idx: int,
    ) -> None:
        """Override in subclasses to handle custom input logic (e.g. keras.Input).

        Args:
            var_name (str): The var_name parameter for the operation.
            node (IRNode): The node parameter for the operation.
            input_prefix (str): The input_prefix parameter for the operation.
            input_idx (int): The input_idx parameter for the operation.
        """
        self.add_line(f"{var_name} = {input_prefix}[{input_idx}]")

    def _emit_output_assignment(
        self,
        node: IRNode,
        input_vars: list[str],
        returns: str,
    ) -> None:
        """Override in subclasses to handle custom output logic.

        Args:
            node (IRNode): The node parameter for the operation.
            input_vars (list[str]): The input_vars parameter for the operation.
            returns (str): The returns parameter for the operation.
        """
        if not hasattr(self, "_output_returns"):
            self._output_returns = []
        self._output_returns.append(returns)

    @classmethod
    def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
        """Eagerly execute an operation.

        Args:
            op_type (str): The operation type.
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: The result.
        """
        raise NotImplementedError("execute_op is not implemented for BaseGenerator")

    @classmethod
    def zeros(cls: type, shape: tuple[int, ...]) -> object:
        """Create a zero tensor.

        Args:
            shape (tuple[int, ...]): The target shape.

        Returns:
            object: The zero tensor.
        """
        raise NotImplementedError("zeros is not implemented for BaseGenerator")

    @classmethod
    def array(cls: type, data: object, dtype: object = None) -> object:
        """Create a tensor array.

        Args:
            data (object): The data to convert.
            dtype (object): The data type.

        Returns:
            object: The tensor array.
        """
        raise NotImplementedError("array is not implemented for BaseGenerator")

    @classmethod
    def asarray(cls: type, data: object) -> object:
        """Convert to tensor array.

        Args:
            data (object): The data parameter for the operation.

        Returns:
            object: The tensor array.
        """
        raise NotImplementedError("asarray is not implemented for BaseGenerator")

    @classmethod
    def item(cls: type, data: object) -> float:
        """Extract item.

        Args:
            data (object): The tensor data.

        Returns:
            float: The scalar value.
        """
        raise NotImplementedError("item is not implemented for BaseGenerator")


class PythonStringGenerator(BaseGenerator):
    """Mixin for python string generators to avoid DRY issues in generate()."""

    _import_header: Union[str, tuple[str, ...]] = ""
    _func_name: str = "evaluate"

    def generate(self) -> str:
        """Generate the complete script."""
        self.code = [self.header]
        if isinstance(self._import_header, str):
            self.add_line(self._import_header)
        elif isinstance(self._import_header, (tuple, list)):
            self.add_line("\n".join(self._import_header))
        self.add_line("")
        self.add_line(f"def {self._func_name}(args):")
        self.indent_level += 1

        self._generate_body("args")

        self.indent_level -= 1
        return "\n".join(self.code)


class ClassBasedGenerator(BaseGenerator):
    """Mixin for class-based string generators to avoid DRY issues in generate()."""

    _forward_method_name: str = "forward"

    def _get_prefix_code(self) -> list[str]:
        """Return the code to be inserted before the class definition."""
        return []

    def _emit_init_body(self) -> bool:
        """Emit initialization code. Return True if params were emitted, False otherwise."""
        return False

    def generate(self) -> str:
        """Generate the complete script."""
        self.code = [self.header]
        self.code.extend(self._get_prefix_code())
        self.add_line("class CompiledModel(nn.Module):")

        # __init__
        self.indent_level = 1
        self.add_line("def __init__(self):")
        self.indent_level += 1
        self.add_line("super().__init__()")

        has_params = self._emit_init_body()
        if not has_params:
            self.add_line("pass")

        self.add_line("")
        self.indent_level -= 1

        # forward
        self.add_line(f"def {self._forward_method_name}(self, *args, **kwargs):")
        self.indent_level += 1

        self._generate_body()

        return "\n".join(self.code)
