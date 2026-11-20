"""Mixins for code generators."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
    EagerBackendAdapter,
    StaticEagerBackendAdapter,
    adapt_backend_module,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir import IRNode


class _GeneratorVisitProto(typing.Protocol):
    """Protocol for `GeneratorLifecycleMixin.visit` method implementations."""

    def __call__(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Visit a logical IR node and return generated code.

        Args:
            node (IRNode): Target IR node.
            input_vars (list[str]): Resolved input variable names.
            **kwargs (BackendArg): Additional keyword attributes.

        Returns:
            str: Emitted code string.
        """
        ...


@typing.runtime_checkable
class _GetModuleClassProtocol(typing.Protocol):
    """Protocol for generator classes exposing `get_module()`."""

    def get_module(self) -> types.ModuleType | BackendModuleProtocol:
        """Return underlying backend module.

        Returns:
            types.ModuleType | BackendModuleProtocol: Backend module.
        """
        ...


class GeneratorLifecycleMixin:
    """Provide mixin for the generation lifecycle."""

    header: str
    code: list[str]
    indent_level: int
    input_idx: int
    _output_returns: list[str] | None
    sorted_nodes: list[IRNode]
    var_names: dict[str, str]
    add_line: Callable[[str], None]
    visit: _GeneratorVisitProto
    emit_constant: Callable[[IRNode], str]
    _emit_body_return: Callable[[list[str]], None]
    _emit_constant_assignment: Callable[[str, str], None]
    _emit_input_assignment: Callable[[str, IRNode, str, int], None]
    _emit_output_assignment: Callable[[IRNode, list[str], str], None]

    def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
        """Assign a deterministic variable name for an IR node.

        Args:
            node_id (str): Target IR node identifier.
            prefix (str): Variable name prefix.

        Returns:
            str: Assigned variable name.
        """
        return f"{prefix}_{node_id}"

    def generate(self) -> str:
        """Generate code from the IR graph.

        Returns:
            str: The generated Python code
        """
        self.code = self._generate_file_header() + self._resolve_imports()
        self._generate_function_signature()
        self._traverse_ir_graph()
        self._generate_return_block()
        return "\n".join(self.code)

    def _generate_file_header(self) -> list[str]:
        """Generate file header with module docstrings.

        Returns:
            list[str]: The header lines.
        """
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Resolve and register required imports.

        Returns:
            list[str]: The import lines.
        """
        return []

    def _generate_function_signature(self) -> None:
        """Generate the main function signature."""
        self.indent_level = 0
        self.add_line("def apply_model(params, *args, **kwargs) -> object:")
        self.indent_level += 1

    def _traverse_ir_graph(self) -> None:
        """Core iteration loop that traverses the IR graph."""
        self._generate_body()

    def _generate_return_block(self) -> None:
        """Format the final return statement (delegated to visitor).

        Returns:
            None
        """
        return None

    def _generate_body(self, input_prefix: str = "args") -> None:
        """Visit nodes to generate code body.

        Args:
            input_prefix (str): The input_prefix parameter for the operation.
        """
        from ml_switcheroo_compiler.backends.base_generator import IRGraphWalker

        walker = IRGraphWalker(self)
        walker.walk(input_prefix)


class EagerExecutionMixin:
    """Provide mixin for eager execution classmethods."""

    @classmethod
    def get_eager_adapter(cls: type) -> EagerBackendAdapter:
        """Return the `EagerBackendAdapter` bound for this backend generator.

        Returns:
            EagerBackendAdapter: Bound eager backend adapter.
        """
        bname = cls.__name__.lower()
        if isinstance(cls, _GetModuleClassProtocol):
            try:
                return adapt_backend_module(cls.get_module(), backend_name=bname)
            except (ImportError, AttributeError, RuntimeError):
                return StaticEagerBackendAdapter(backend_name=bname)
        return StaticEagerBackendAdapter(backend_name=bname)

    @classmethod
    def execute_op(cls: type, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute an operation eagerly.

        Args:
            op_type (str): The operation type.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            BackendArg: The eager evaluation result.
        """
        del args, kwargs
        return None

    @classmethod
    def zeros(cls: type, shape: tuple[int, ...]) -> BackendArg:
        """Execute zeros for compiler processing.

        Args:
            shape (tuple[int, ...]): Input `shape` argument (shape).

        Returns:
            BackendArg: Computed `BackendArg` output.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_zeros
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        mod = cls.get_module() if isinstance(cls, _GetModuleClassProtocol) else get_active_backend().get_module()
        return generic_zeros(mod, shape)

    @classmethod
    def array(cls: type, data: BackendArg, dtype: BackendArg | None = None) -> BackendArg:
        """Execute array for compiler processing.

        Args:
            data: Input `data` argument (data).
            dtype: Input `dtype` argument (dtype).

        Returns:
            BackendArg: Computed `BackendArg` output.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_array
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        mod = cls.get_module() if isinstance(cls, _GetModuleClassProtocol) else get_active_backend().get_module()
        return generic_array(mod, data, dtype)

    @classmethod
    def asarray(cls: type, data: BackendArg) -> BackendArg:
        """Execute asarray for compiler processing.

        Args:
            data: Input `data` argument (data).

        Returns:
            BackendArg: Computed `BackendArg` output.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_asarray
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        mod = cls.get_module() if isinstance(cls, _GetModuleClassProtocol) else get_active_backend().get_module()
        return generic_asarray(mod, data)

    @classmethod
    def item(cls: type, data: BackendArg) -> float:
        """Execute item for compiler processing.

        Args:
            data: Input `data` argument (data).

        Returns:
            float: Computed `float` output.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_item
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        mod = cls.get_module() if isinstance(cls, _GetModuleClassProtocol) else get_active_backend().get_module()
        return generic_item(mod, data)
