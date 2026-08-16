# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Define base generator for emitting backend code from IR."""

import re
from dataclasses import dataclass
from typing import Any, Union

from ml_switcheroo_compiler.backends.formatters import CodeFormatter, FormatterContext, OpFormatter
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

from .generator_mixins import EagerExecutionMixin, GeneratorLifecycleMixin


class IRGraphWalker:
    """Help class to encapsulate IR graph traversal logic."""

    def __init__(self, generator: "BaseGenerator") -> None:
        """Initialize the walker.

        Args:
            generator (BaseGenerator): The generator instance.
        """
        self.generator = generator

    def walk(self, input_prefix: str = "args") -> None:
        """Walk the graph and generate code.

        Args:
            input_prefix (str): The prefix for input args.
        """
        visitor = CodeGeneratorVisitor(self.generator)
        visitor.generate_body(input_prefix)


@dataclass
class InputContext:
    """Provide context for input assignment."""

    var_name: str
    node: IRNode
    input_prefix: str
    input_idx: int


class FormatterProxyMixin:
    """FormatterProxyMixin class."""

    formatter: Any
    _formatter: Any
    """Provide mixin for proxying formatter methods."""

    @property
    def var_names(self) -> dict[str, str]:
        """Provide proxy property for formatter var_names.

        Returns:
            dict[str, str]: The variable names map.
        """
        return self.formatter.var_names  # type: ignore

    @var_names.setter
    def var_names(self, value: dict[str, str]) -> None:
        """Set formatter var_names.

        Args:
            value (dict[str, str]): The variable names map.
        """
        self.formatter.var_names = value

    @property
    def code(self) -> list[str]:
        """Provide proxy property for formatter code.

        Returns:
            list[str]: The generated code list.
        """
        return self.formatter.code  # type: ignore

    @code.setter
    def code(self, value: list[str]) -> None:
        """Set formatter code.

        Args:
            value (list[str]): The generated code list.
        """
        self.formatter.code = value

    @property
    def indent_level(self) -> int:
        """Provide proxy property for formatter indent_level.

        Returns:
            int: The current indent level.
        """
        return self.formatter.indent_level  # type: ignore

    @indent_level.setter
    def indent_level(self, value: int) -> None:
        """Set formatter indent_level.

        Args:
            value (int): The current indent level.
        """
        self.formatter.indent_level = value

    @property
    def header(self) -> str:
        """Provide proxy property for formatter header.

        Returns:
            str: The header string.
        """
        return self.formatter.header  # type: ignore

    @header.setter
    def header(self, value: str) -> None:
        """Set formatter header.

        Args:
            value (str): The header string.
        """
        self.formatter.header = value

    def get_indent(self) -> str:
        """Get the indentation string.

        Returns:
            str: Indentation string.
        """
        return self.formatter.get_indent()  # type: ignore

    def add_line(self: Any, line: str) -> None:
        """Add a line of code.

        Args:
            line (str): The line to add.
        """
        self.formatter.add_line(line)

    def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
        """Assign a variable name.

        Args:
            node_id (str): The node ID.
            prefix (str): The prefix to use.

        Returns:
            str: The assigned variable name.
        """
        return self.formatter.assign_var_name(node_id, prefix)  # type: ignore


class EmitUtilsMixin:
    """EmitUtilsMixin class."""

    add_line: Any
    """Provide mixin for emit utilities."""

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

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Override in subclasses to emit framework-specific constant arrays.

        Args:
            var_name (str): The var_name parameter for the operation.
            val_repr (str): The val_repr parameter for the operation.
        """
        self.add_line(f"{var_name} = {val_repr}")


class BaseGenerator(FormatterProxyMixin, EmitUtilsMixin, GeneratorLifecycleMixin, EagerExecutionMixin):
    """Abstract base class for backend code generation."""

    def __init__(self, graph: IRGraph, delegates: Any = None) -> None:
        """Initialize the object.

        Args:
            graph (IRGraph): The graph to process.
            delegates: The visitor delegates.
        """
        self.graph = graph
        self.sorted_nodes = topological_sort(graph)
        self.formatter = CodeFormatter()
        self.visitors = [self] + (delegates or [])

    def emit_constant(self, node: IRNode) -> str:
        """Emit code for the constant backend.

        Args:
            node (IRNode): The node to process.

        Returns:
            str: The computed result.
        """
        val = node.attributes.get("value")
        return repr(val)

    def visit(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Visit a node and return the formatted code string for the operation.

        Args:
            node (IRNode): The node parameter for the operation.
            input_vars (list[str]): The input_vars parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The computed result.
        """
        op_type = getattr(node, "op_type", "")
        method_name = f"visit_{op_type}"
        for visitor in getattr(self, "visitors", []):
            if hasattr(visitor, method_name):
                method = getattr(visitor, method_name)
                return method(node, input_vars, **kwargs)  # type: ignore
        return self.generic_visit(node, input_vars, **kwargs)

    def get_ops_map(self, kwargs: dict[str, Any]) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            Dictionary mapping operation type to format string.
        """
        from ml_switcheroo_compiler.ops.registry import backend_mapping_registry

        ops = {}
        prefix = self.get_fallback_prefix()
        for op_name in backend_mapping_registry.operations.keys():
            fmt = backend_mapping_registry.get_generator_mapping(prefix, op_name)
            if fmt is not None:
                ops[op_name] = fmt

        if "OverlapAndAdd" not in ops:
            ops["OverlapAndAdd"] = "tf.signal.overlap_and_add({0})"

        return ops

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations.

        Returns:
            str: The prefix.
        """
        return "np"

    def get_fallback_axis_kwarg(self) -> str:
        """Get the fallback axis keyword argument name.

        Returns:
            str: The axis keyword.
        """
        return "axis"

    def get_fallback_keepdims_kwarg(self) -> str:
        """Get the fallback keepdims keyword argument name.

        Returns:
            str: The keepdims keyword.
        """
        return "keepdims"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Fallback visit method for operations not explicitly handled.

        Args:
            node (IRNode): The node.
            input_vars (list[str]): The inputs.
            **kwargs: Additional attributes.

        Returns:
            str: The code string.
        """
        op_type = getattr(node, "op_type", "")
        ops_map = self.get_ops_map(kwargs)
        if op_type in ops_map:
            fmt = ops_map[op_type]
            fmt = OpFormatter.format_backend_string(fmt, input_vars, kwargs)
            fmt = re.sub(", \\w+=\\{[^\\}]+\\}", "", fmt)
            return fmt

        ctx = FormatterContext(
            prefix=self.get_fallback_prefix(),
            op_type=op_type,
            input_vars=input_vars,
            kwargs=kwargs,
            axis_kwarg=self.get_fallback_axis_kwarg(),
            keepdims_kwarg=self.get_fallback_keepdims_kwarg(),
        )
        return OpFormatter.format_generic_fallback(ctx)

    def _emit_input_assignment(self, var_name: str, node: IRNode, input_prefix: str, input_idx: int) -> None:
        """Override in subclasses to handle custom input logic (e.g. keras.Input).

        Args:
            var_name (str): The var_name parameter for the operation.
            node (IRNode): The node parameter for the operation.
            input_prefix (str): The input_prefix parameter for the operation.
            input_idx (int): The input_idx parameter for the operation.
        """
        self.add_line(f"{var_name} = {input_prefix}[{input_idx}]")

    def _emit_output_assignment(self, node: IRNode, input_vars: list[str], returns: str) -> None:
        """Override in subclasses to handle custom output logic.

        Args:
            node (IRNode): The node parameter for the operation.
            input_vars (list[str]): The input_vars parameter for the operation.
            returns (str): The returns parameter for the operation.
        """
        if not hasattr(self, "_output_returns"):
            self._output_returns = []
        self._output_returns.append(returns)


class PythonStringGenerator(BaseGenerator):
    """Provide mixin for python string generators to avoid DRY issues in generate()."""

    _import_header: Union[str, tuple[str, ...]] = ""
    _func_name: str = "evaluate"

    def generate(self) -> str:
        """Generate the complete script.

        Returns:
            str: The generated script.
        """
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
    """ClassBasedGenerator class."""

    get_language: Any
    """Provide mixin for class-based string generators to avoid DRY issues in generate()."""

    _forward_method_name: str = "forward"
    _base_class_name: str = ""

    def _get_prefix_code(self) -> list[str]:
        """Return the code to be inserted before the class definition.

        Returns:
            list[str]: The prefix code lines.
        """
        return []

    def _emit_init_body(self) -> bool:
        """Emit initialization code.

        Returns:
            bool: True if params were emitted, False otherwise.
        """
        return False

    def generate(self) -> str:
        """Generate the complete script.

        Returns:
            str: The generated script.
        """
        self.code = [self.header]
        self.code.extend(self._get_prefix_code())
        base_class = f"({self._base_class_name})" if self._base_class_name else ""
        self.add_line(f"class CompiledModel{base_class}:")
        self.indent_level = 1
        self.add_line("def __init__(self):")
        self.indent_level += 1
        if self._base_class_name:
            self.add_line("super().__init__()")
        has_params = self._emit_init_body()
        if not has_params:
            self.add_line("pass" if self.get_language() == "python" else "")
        self.add_line("")
        self.indent_level -= 1
        self.add_line(f"def {self._forward_method_name}(self, *args, **kwargs):")
        self.indent_level += 1
        self._generate_body()
        return "\n".join(self.code)
