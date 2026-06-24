"""Mixins for code generators."""


class GeneratorLifecycleMixin:
    """Mixin for the generation lifecycle."""

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
        """Generate file header with module docstrings."""
        return [self.header.strip()]  # pragma: no cover

    def _resolve_imports(self) -> list[str]:
        """Resolve and register required imports."""
        return []  # pragma: no cover

    def _generate_function_signature(self) -> None:
        """Generate the main function signature."""
        self.indent_level = 0  # pragma: no cover
        self.add_line("def apply_model(params, *args, **kwargs):")  # pragma: no cover
        self.indent_level += 1  # pragma: no cover

    def _traverse_ir_graph(self) -> None:
        """Core iteration loop that traverses the IR graph."""
        self._generate_body()

    def _generate_return_block(self) -> None:
        """Format the final return statement (delegated to visitor)."""
        pass

    def _generate_body(self, input_prefix: str = "args") -> None:
        """Visit nodes to generate code body.

        Args:
            input_prefix (str): The input_prefix parameter for the operation.
        """
        from ml_switcheroo_compiler.backends.base_generator import IRGraphWalker

        walker = IRGraphWalker(self)
        walker.walk(input_prefix)


class EagerExecutionMixin:
    """Mixin for eager execution classmethods."""

    @classmethod
    def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
        """Function docstring.

        Args:
            op_type: Arg.
            args: Arg.
            kwargs: Arg.
        """
        raise NotImplementedError("BaseGenerator cannot execute ops")  # pragma: no cover

    @classmethod
    def zeros(cls: type, shape: tuple[int, ...]) -> object:
        """Function docstring.

        Args:
            shape: Arg.
        """
        raise NotImplementedError("Zeros not implemented")  # pragma: no cover

    @classmethod
    def array(cls: type, data: object, dtype: object = None) -> object:
        """Function docstring.

        Args:
            data: Arg.
            dtype: Arg.
        """
        raise NotImplementedError("Array not implemented")  # pragma: no cover

    @classmethod
    def asarray(cls: type, data: object) -> object:
        """Function docstring.

        Args:
            data: Arg.
        """
        raise NotImplementedError("Asarray not implemented")  # pragma: no cover

    @classmethod
    def item(cls: type, data: object) -> float:
        """Function docstring.

        Args:
            data: Arg.
        """
        raise NotImplementedError("Item not implemented")  # pragma: no cover
