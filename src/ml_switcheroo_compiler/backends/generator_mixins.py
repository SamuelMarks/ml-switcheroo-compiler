"""Mixins for code generators."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, as_float, as_tensor

if typing.TYPE_CHECKING:
    pass


class GeneratorLifecycleMixin:
    """Provide mixin for the generation lifecycle."""

    header: str
    code: list[str]
    indent_level: int
    add_line: Callable[[str], None]

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
        from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, IRGraphWalker

        walker: IRGraphWalker = IRGraphWalker(typing.cast(BaseGenerator, self))
        walker.walk(input_prefix)


class EagerExecutionMixin:
    """Provide mixin for eager execution classmethods."""

    @classmethod
    def execute_op(cls: type, op_type: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Execute an operation eagerly.

        Args:
            op_type (str): The operation type.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            EagerValue: The eager evaluation result.
        """
        del args, kwargs
        return None

    @classmethod
    def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
        """Execute zeros for compiler processing.

        Args:
            shape (tuple[int, ...]): The shape parameter.

        Returns:
            EagerValue: Result.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_zeros
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        mod = cls.get_module() if hasattr(cls, "get_module") else BackendRegistry.get("numpy").get_module()
        return generic_zeros(mod, shape)

    @classmethod
    def array(cls: type, data: EagerValue, dtype: EagerValue | None = None) -> EagerValue:
        """Execute array for compiler processing.

        Args:
            data: The data parameter.
            dtype: The dtype parameter.

        Returns:
            EagerValue: Result.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_array
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        mod = cls.get_module() if hasattr(cls, "get_module") else BackendRegistry.get("numpy").get_module()
        return as_tensor(as_callable(generic_array)(mod, data, dtype))

    @classmethod
    def asarray(cls: type, data: EagerValue) -> EagerValue:
        """Execute asarray for compiler processing.

        Args:
            data: The data parameter.

        Returns:
            EagerValue: Result.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_asarray
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        mod = cls.get_module() if hasattr(cls, "get_module") else BackendRegistry.get("numpy").get_module()
        return as_tensor(as_callable(generic_asarray)(mod, data))

    @classmethod
    def item(cls: type, data: EagerValue) -> float:
        """Execute item for compiler processing.

        Args:
            data: The data parameter.

        Returns:
            float: Result.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_item
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        mod = cls.get_module() if hasattr(cls, "get_module") else BackendRegistry.get("numpy").get_module()
        return as_float(as_callable(generic_item)(mod, data))
