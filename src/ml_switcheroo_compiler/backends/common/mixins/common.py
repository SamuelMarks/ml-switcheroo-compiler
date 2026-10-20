"""Implementation of common operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.common.mixins.common`)."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import AttributeValue

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.backends.formatters import CodeFormatter


@typing.runtime_checkable
class FallbackPrefixProtocol(typing.Protocol):
    """Protocol for AST generators providing a fallback prefix."""

    indent_level: int
    var_names: dict[str, str]
    formatter: CodeFormatter
    _cholesky_vjp_imported: bool
    _lu_vjp_imported: bool
    _qr_vjp_imported: bool
    _svd_vjp_imported: bool

    @property
    def code(self) -> list[str]:
        """Return generated code lines."""
        ...

    def __init__(self, graph: AttributeValue = None) -> None:
        """Initialize the protocol generator.

        Args:
            graph (AttributeValue): Optional computation graph.
        """
        ...

    def get_fallback_prefix(self) -> str:
        """Return the backend prefix.

        Returns:
            str: Fallback prefix string.
        """
        ...

    def add_line(self, line: str) -> None:
        """Add a line of code.

        Args:
            line (str): The line of code to add.
        """
        ...

    def add_import(self, imp: str) -> None:
        """Register an import statement.

        Args:
            imp (str): The import statement to register.
        """
        ...


class CommonASTVisitor:
    """Define base mixin/visitor for shared AST generation logic across backends."""

    indent_level: int
    code: list[str]
    var_names: dict[str, str]
    _cholesky_vjp_imported: bool
    _lu_vjp_imported: bool
    _qr_vjp_imported: bool
    _svd_vjp_imported: bool

    def __init__(
        self,
        graph: AttributeValue = None,
        *args: str | int | float | bool,
        generator: FallbackPrefixProtocol | BaseGenerator | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Initialize the visitor.

        Args:
            graph (AttributeValue): Optional computation graph.
            *args (str | int | float | bool): Positional arguments.
            generator (FallbackPrefixProtocol | None): The delegate generator.
            **kwargs (str | int | float | bool | None): Keyword arguments.
        """
        self._generator = generator
        del graph, args, kwargs
        super().__init__()

    @property
    def generator(self) -> FallbackPrefixProtocol:
        """Get the delegate generator.

        Returns:
            FallbackPrefixProtocol: The generator or self.
        """
        return typing.cast(FallbackPrefixProtocol, getattr(self, "_generator", None) or self)

    def get_fallback_prefix(self) -> str:
        """Return the backend prefix (e.g., 'jax', 'pt', 'mx').

        Returns:
            str: Result.
        """
        return ""
