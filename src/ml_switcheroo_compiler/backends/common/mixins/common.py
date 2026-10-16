"""Provide common AST visitor mixin module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from ml_switcheroo_ir import AttributeValue

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.formatters import CodeFormatter


@runtime_checkable
class FallbackPrefixProtocol(Protocol):
    """Protocol for AST generators providing a fallback prefix."""

    indent_level: int
    code: list[str]
    var_names: dict[str, str]
    formatter: CodeFormatter
    _cholesky_vjp_imported: bool
    _lu_vjp_imported: bool
    _qr_vjp_imported: bool
    _svd_vjp_imported: bool

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
        generator: FallbackPrefixProtocol | None = None,
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
        return cast(FallbackPrefixProtocol, getattr(self, "_generator", None) or self)

    def get_fallback_prefix(self) -> str:
        """Return the backend prefix (e.g., 'jax', 'pt', 'mx').

        Returns:
            str: Result.
        """
        return ""
