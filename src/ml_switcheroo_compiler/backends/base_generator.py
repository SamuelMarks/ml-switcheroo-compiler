"""Define base generator for emitting backend code from IR."""

from __future__ import annotations

import typing  # noqa: F401
from collections.abc import Sequence
from types import ModuleType

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_str,
)
from ml_switcheroo_compiler.core.shape import OpValue as OpValue
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends._base_generator_runtime import (
    BaseGeneratorRuntimeMixin as BaseGeneratorRuntimeMixin,
)
from ml_switcheroo_compiler.backends._base_generator_runtime import (
    _CompileAOTMethodProtocol as _CompileAOTMethodProtocol,
)
from ml_switcheroo_compiler.backends._base_generator_runtime import (
    class_or_instance_method as class_or_instance_method,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    CleanableResourceProtocol as CleanableResourceProtocol,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    CompiledArtifact as CompiledArtifact,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    EmitUtilsMixin as EmitUtilsMixin,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    FormatterProxyMixin as FormatterProxyMixin,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    InputContext as InputContext,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    IRGraphWalker as IRGraphWalker,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    VisitorProtocol as VisitorProtocol,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    _extract_node_attributes as _extract_node_attributes,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    _GraphHolderProtocol as _GraphHolderProtocol,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    _NodeAttributesProtocol as _NodeAttributesProtocol,
)
from ml_switcheroo_compiler.backends._base_generator_support import (
    _resolve_visitor_method as _resolve_visitor_method,
)
from ml_switcheroo_compiler.backends._base_generator_templates import (
    BaseGeneratorTemplateMixin as BaseGeneratorTemplateMixin,
)
from ml_switcheroo_compiler.backends.formatters import CodeFormatter
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor as CodeGeneratorVisitor
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

from .generator_mixins import EagerExecutionMixin, GeneratorLifecycleMixin


class BaseGenerator(
    BaseGeneratorTemplateMixin,
    BaseGeneratorRuntimeMixin,
    FormatterProxyMixin,
    EmitUtilsMixin,
    GeneratorLifecycleMixin,
    EagerExecutionMixin,
):
    """Abstract base class for backend code generation."""

    input_idx: int
    module: ModuleType
    get_numpy_rng: TypedVariadicCallable[TensorLike]
    numpy: TypedVariadicCallable[BackendArg]
    backend_name: str | None = None
    graph: IRGraph = IRGraph()
    visitors: list[VisitorProtocol] = []
    _aot_cache: dict[tuple[int, tuple[tuple[str, str], ...]], CompiledArtifact] | None = None
    _output_returns: list[str] | None = None
    _cholesky_vjp_imported: bool = False
    _lu_vjp_imported: bool = False
    _qr_vjp_imported: bool = False
    _svd_vjp_imported: bool = False

    if typing.TYPE_CHECKING:

        @classmethod
        def get_module(cls) -> ModuleType:
            """Return the underlying backend Python module.

            Returns:
                ModuleType: Backend module.
            """
            ...

        @classmethod
        def get_scipy_module(cls, submodule: str = "special") -> ModuleType:
            """Return the underlying SciPy module.

            Args:
                submodule (str): Submodule name.

            Returns:
                ModuleType: SciPy submodule.
            """
            ...

        @classmethod
        def load(
            cls,
            filepath: str,
            allow_pickle: bool = False,
            fix_imports: bool = True,
            encoding: str = "ASCII",
        ) -> BackendArg:
            """Load serialized array or checkpoint from filepath.

            Args:
                filepath (str): Path to file.
                allow_pickle (bool): Allow pickled objects.
                fix_imports (bool): Fix Python 2 imports.
                encoding (str): String encoding.

            Returns:
                BackendArg: Loaded array or dictionary.
            """
            ...

    def __init__(
        self,
        graph: IRGraph,
        delegates: Sequence[VisitorProtocol] | Sequence[CodeGeneratorVisitor] | BackendArg = None,
    ) -> None:
        """Initialize the object.

        Args:
            graph (IRGraph): The graph to process.
            delegates (Sequence[VisitorProtocol] | Sequence[CodeGeneratorVisitor] | BackendArg): The visitor delegates.
        """
        self.graph = graph
        self.sorted_nodes = topological_sort(graph)
        self.formatter = CodeFormatter()
        delegate_items: list[VisitorProtocol] = (
            [d for d in delegates if isinstance(d, VisitorProtocol)] if isinstance(delegates, (list, tuple)) else []
        )
        self.visitors: list[VisitorProtocol] = [self] + delegate_items

    def get_language(self) -> str:
        """Get the target programming language for code emission.

        Returns:
            str: Target language string ('python').
        """
        return "python"

    def emit_constant(self, node: IRNode) -> str:
        """Emit code for the constant backend.

        Args:
            node (IRNode): The node to process.

        Returns:
            str: The computed result.
        """
        val = node.attributes.get("value")
        return repr(val)

    def visit_OpNode(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Generic visitor for IRNode driven by declarative mapping YAML templates.

        Args:
            node (IRNode): Target IRNode.
            input_vars (list[str]): Input variable names.
            **kwargs (BackendArg): Additional node attributes and context.

        Returns:
            str: Generated code line or expression.
        """
        backend_name = self.backend_name or self.get_fallback_prefix()
        op_name = node.op_type
        ast_template, kwarg_map, target_api = self._load_op_mapping_info(backend_name, op_name)

        all_attrs: dict[str, BackendArg] = dict(_extract_node_attributes(node))
        all_attrs.update(kwargs)
        translated_kwargs = self._translate_node_kwargs(all_attrs, kwarg_map)

        if ast_template:
            return self._format_ast_template(ast_template, node, input_vars, all_attrs, translated_kwargs)

        if target_api:
            call_args = list(input_vars)
            kw_pairs = [f"{k}={v}" for k, v in translated_kwargs.items() if not str(k).startswith("_")]
            call_str = ", ".join(call_args + kw_pairs)
            return f"{target_api}({call_str})"

        return self.generic_visit(node, input_vars, **translated_kwargs)

    def visit(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Visit a node and return the formatted code string for the operation.

        Args:
            node (IRNode): The node parameter for the operation.
            input_vars (list[str]): The input_vars parameter for the operation.
            **kwargs (BackendArg): Additional keyword arguments.

        Returns:
            str: The computed result.
        """
        op_type = node.op_type
        method_name = f"visit_{op_type}"
        for visitor in self.visitors:
            method = _resolve_visitor_method(visitor, method_name)
            if method is not None:
                return coerce_str(method(node, input_vars, **kwargs))
        self_method = _resolve_visitor_method(self, method_name) if method_name != "visit_OpNode" else None
        if self_method is not None:
            return coerce_str(self_method(node, input_vars, **kwargs))
        op_node_method = _resolve_visitor_method(self, "visit_OpNode")
        if op_node_method is not None:
            return coerce_str(op_node_method(node, input_vars, **kwargs))
        return self.generic_visit(node, input_vars, **kwargs)

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
        if self._output_returns is None:
            self._output_returns = []
        self._output_returns.append(returns)


class PythonStringGenerator(BaseGenerator):
    """Provide mixin for python string generators to avoid DRY issues in generate()."""

    _import_header: str | tuple[str, ...] = ""
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
        self.add_line("def __init__(self, *args, **kwargs) -> None:")
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
