"""Base generator for emitting backend code from IR."""

import re
from dataclasses import dataclass
from typing import Union

from ml_switcheroo_compiler.backends.formatters import CodeFormatter, FormatterContext, OpFormatter
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

from .generator_mixins import EagerExecutionMixin, GeneratorLifecycleMixin


class IRGraphWalker:
    """Helper class to encapsulate IR graph traversal logic."""

    def __init__(self, generator: "BaseGenerator") -> None:
        """Init walker."""
        self.generator = generator

    def walk(self, input_prefix: str = "args") -> None:
        """Walk the graph."""
        visitor = CodeGeneratorVisitor(self.generator)
        visitor.generate_body(input_prefix)


@dataclass
class InputContext:
    """Context for input assignment."""

    var_name: str
    node: IRNode
    input_prefix: str
    input_idx: int


class FormatterProxyMixin:
    """Mixin for proxying formatter methods."""

    @property
    def var_names(self) -> dict[str, str]:
        """Proxy property for formatter var_names."""
        return self.formatter.var_names

    @var_names.setter
    def var_names(self, value: dict[str, str]) -> None:
        """Proxy property for formatter var_names."""
        self.formatter.var_names = value  # pragma: no cover

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
        self.formatter.header = value  # pragma: no cover

    def get_indent(self) -> str:
        """Evaluate get indent."""
        return self.formatter.get_indent()  # pragma: no cover

    def add_line(self, line: str) -> None:
        """Add line."""
        self.formatter.add_line(line)

    def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
        """Assign var name."""
        return self.formatter.assign_var_name(node_id, prefix)


class EmitUtilsMixin:
    """Mixin for emit utilities."""

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

    def __init__(self, graph: IRGraph, delegates: list = None) -> None:
        """Initializes the object.

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
            str: The evaluated output resulting from this operation.
        """
        val = node.attributes.get("value")
        return repr(val)

    def visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
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
        for visitor in getattr(self, "visitors", []):
            if hasattr(visitor, method_name):
                method = getattr(visitor, method_name)
                return method(node, input_vars, **kwargs)
        return self.generic_visit(node, input_vars, **kwargs)

    def _get_math_ops(self, kwargs: dict) -> dict[str, str]:
        """Function docstring."""
        return {}

    def _get_linalg_ops(self, kwargs: dict) -> dict[str, str]:
        """Function docstring."""
        return {}

    def _get_nn_ops(self, kwargs: dict) -> dict[str, str]:
        """Function docstring."""
        return {}

    def _get_creation_ops(self, kwargs: dict) -> dict[str, str]:
        """Function docstring."""
        return {}

    def _get_array_ops(self, kwargs: dict) -> dict[str, str]:
        """Function docstring."""
        return {}

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            Dictionary mapping operation type to format string.
        """
        ops = {}
        ops.update(self._get_math_ops(kwargs))
        ops.update(self._get_linalg_ops(kwargs))
        ops.update(self._get_nn_ops(kwargs))
        ops.update(self._get_creation_ops(kwargs))
        ops.update(self._get_array_ops(kwargs))

        # Audio/Signal ops shared defaults (usually mapping to tf.signal as a placeholder/fallback)
        ops.update(
            {
                "Dct": "tf.signal.dct({0})",
                "Idct": "tf.signal.idct({0})",
                "Mdct": "tf.signal.mdct({0})",
                "InverseMdct": "tf.signal.inverse_mdct({0})",
                "Frame": "tf.signal.frame({0})",
                "OverlapAndAdd": "tf.signal.overlap_and_add({0})",
                "BandedTriangularSolve": "tf.linalg.banded_triangular_solve",
                "EighTridiagonal": "tf.linalg.eigh_tridiagonal",
                "MatrixRank": "tf.linalg.matrix_rank",
                "MatrixTranspose": "tf.linalg.matrix_transpose",
                "Sqrtm": "tf.linalg.sqrtm",
            }
        )

        return ops

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations."""
        return "np"  # pragma: no cover

    def get_fallback_axis_kwarg(self) -> str:
        """Get the fallback axis keyword argument name."""
        return "axis"

    def get_fallback_keepdims_kwarg(self) -> str:
        """Get the fallback keepdims keyword argument name."""
        return "keepdims"

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
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
            self._output_returns = []  # pragma: no cover
        self._output_returns.append(returns)


class PythonStringGenerator(BaseGenerator):
    """Mixin for python string generators to avoid DRY issues in generate()."""

    _import_header: Union[str, tuple[str, ...]] = ""
    _func_name: str = "evaluate"

    def generate(self) -> str:
        """Generate the complete script."""
        self.code = [self.header]
        if isinstance(self._import_header, str):
            self.add_line(self._import_header)
        elif isinstance(self._import_header, (tuple, list)):  # pragma: no cover
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
        return []  # pragma: no cover

    def _emit_init_body(self) -> bool:
        """Emit initialization code. Return True if params were emitted, False otherwise."""
        return False

    def generate(self) -> str:
        """Generate the complete script."""
        self.code = [self.header]
        self.code.extend(self._get_prefix_code())
        self.add_line("class CompiledModel(nn.Module):")
        self.indent_level = 1
        self.add_line("def __init__(self):")
        self.indent_level += 1
        self.add_line("super().__init__()")
        has_params = self._emit_init_body()
        if not has_params:
            self.add_line("pass")
        self.add_line("")
        self.indent_level -= 1
        self.add_line(f"def {self._forward_method_name}(self, *args, **kwargs):")
        self.indent_level += 1
        self._generate_body()
        return "\n".join(self.code)
