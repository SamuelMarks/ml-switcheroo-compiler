"""Operations and utilities for `ml_switcheroo_compiler.backends.common.mixins.common`."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.formatters import CodeFormatter
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class FallbackPrefixProtocol(typing.Protocol):
    """Protocol for AST generators providing a fallback prefix."""

    indent_level: int
    var_names: dict[str, str]
    _cholesky_vjp_imported: bool
    _lu_vjp_imported: bool
    _qr_vjp_imported: bool
    _svd_vjp_imported: bool

    @property
    def code(self) -> list[str]:
        """Return generated code lines.

        Returns:
            list[str]: Code lines.
        """
        ...

    def __init__(self, graph: BackendArg = None) -> None:
        """Initialize the protocol generator.

        Args:
            graph (BackendArg): Optional computation graph.
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


class CommonASTVisitor:
    """Define base mixin/visitor for shared AST generation logic across backends."""

    indent_level: int = 0
    _cholesky_vjp_imported: bool = False
    _lu_vjp_imported: bool = False
    _qr_vjp_imported: bool = False
    _svd_vjp_imported: bool = False
    _generator: FallbackPrefixProtocol | None = None
    add_line: Callable[[str], None]

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
        if generator is None and not adapt_backend_module(self).resolve_symbol("formatter")[0]:
            fmt = CodeFormatter()
            self.var_names: dict[str, str] = fmt.var_names
            self.code: list[str] = fmt.code
        del graph, args, kwargs
        super().__init__()

    @property
    def generator(self) -> FallbackPrefixProtocol:
        """Get the delegate generator.

        Returns:
            FallbackPrefixProtocol: The generator or self.
        """
        return self._generator if self._generator is not None else self

    def get_fallback_prefix(self) -> str:
        """Return the backend prefix (e.g., 'jax', 'pt', 'mx').

        Returns:
            str: Computed `str` output.
        """
        return ""
