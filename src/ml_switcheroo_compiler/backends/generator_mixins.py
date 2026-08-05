# ruff: noqa: E501
"""Mixins for code generators."""


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
        self.add_line("def apply_model(params, *args, **kwargs):")
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
    def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
        """Execute an operation eagerly.

        Args:
            op_type (str): The operation type.
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: The eager evaluation result.
        """
        return None

    @classmethod
    def zeros(cls: type, shape: tuple[int, ...]) -> object:
        """Evaluate zeros operation.

        Args:
        shape (object): The shape parameter.

        Returns:
        object: Result.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_zeros

        return generic_zeros(cls.get_module() if hasattr(cls, "get_module") else __import__("numpy"), shape)

    @classmethod
    def array(cls: type, data: object, dtype: object = None) -> object:
        """Evaluate array operation.

        Args:
        data (object): The data parameter.
        dtype (object): The dtype parameter.

        Returns:
        object: Result.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_array

        return generic_array(cls.get_module() if hasattr(cls, "get_module") else __import__("numpy"), data, dtype)

    @classmethod
    def asarray(cls: type, data: object) -> object:
        """Evaluate asarray operation.

        Args:
        data (object): The data parameter.

        Returns:
        object: Result.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_asarray

        return generic_asarray(cls.get_module() if hasattr(cls, "get_module") else __import__("numpy"), data)

    @classmethod
    def item(cls: type, data: object) -> float:
        """Evaluate item operation.

        Args:
        data (object): The data parameter.

        Returns:
        float: Result.
        """
        from ml_switcheroo_compiler.backends.eager.types_utils import generic_item

        return generic_item(cls.get_module() if hasattr(cls, "get_module") else __import__("numpy"), data)
