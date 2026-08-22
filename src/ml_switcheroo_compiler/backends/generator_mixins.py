# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Mixins for code generators."""

from typing import Any


class GeneratorLifecycleMixin:
    """Provide mixin for the generation lifecycle."""

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
        return [self.header.strip()]  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def _resolve_imports(self) -> list[str]:
        """Resolve and register required imports.

        Returns:
            list[str]: The import lines.
        """
        return []

    def _generate_function_signature(self) -> None:
        """Generate the main function signature."""
        self.indent_level = 0
        self.add_line("def apply_model(params, *args, **kwargs):")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
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

        walker = IRGraphWalker(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        walker.walk(input_prefix)


class EagerExecutionMixin:
    """Provide mixin for eager execution classmethods."""

    @classmethod
    def execute_op(cls: type, op_type: str, *args: Any, **kwargs: Any) -> Any:
        """Execute an operation eagerly.

        Args:
            op_type (str): The operation type.
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: The eager evaluation result.
        """
        return None

    @classmethod
    def zeros(cls: type, shape: tuple[int, ...]) -> Any:
        """Evaluate zeros operation.

        Args:
        shape (object): The shape parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        import numpy

        from ml_switcheroo_compiler.backends.eager.types_utils import generic_zeros

        return generic_zeros(cls.get_module() if hasattr(cls, "get_module") else numpy, shape)

    @classmethod
    def array(cls: type, data: Any, dtype: Any = None) -> Any:
        """Evaluate array operation.

        Args:
        data (object): The data parameter.
        dtype (object): The dtype parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        import numpy

        from ml_switcheroo_compiler.backends.eager.types_utils import generic_array

        return generic_array(cls.get_module() if hasattr(cls, "get_module") else numpy, data, dtype)

    @classmethod
    def asarray(cls: type, data: Any) -> Any:
        """Evaluate asarray operation.

        Args:
        data (object): The data parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        import numpy

        from ml_switcheroo_compiler.backends.eager.types_utils import generic_asarray

        return generic_asarray(cls.get_module() if hasattr(cls, "get_module") else numpy, data)

    @classmethod
    def item(cls: type, data: Any) -> float:
        """Evaluate item operation.

        Args:
        data (object): The data parameter.

        Returns:
        float: Result.
        """
        import numpy

        from ml_switcheroo_compiler.backends.eager.types_utils import generic_item

        return generic_item(cls.get_module() if hasattr(cls, "get_module") else numpy, data)  # type: ignore
