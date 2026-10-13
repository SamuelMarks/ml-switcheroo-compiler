"""Provide common AST visitor mixin module."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# ruff: noqa: E402, E501, F403


@runtime_checkable
class FallbackPrefixProtocol(Protocol):
    """Protocol for AST generators providing a fallback prefix."""

    def get_fallback_prefix(self) -> str:
        """Return the backend prefix.

        Returns:
            str: Fallback prefix string.
        """
        ...


class CommonASTVisitor:
    """Define base mixin/visitor for shared AST generation logic across backends."""

    def __init__(
        self,
        *args: str | int | float | bool,
        generator: FallbackPrefixProtocol | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Initialize the visitor.

        Args:
            *args (str | int | float | bool): Positional arguments.
            generator (FallbackPrefixProtocol | None): The delegate generator.
            **kwargs (str | int | float | bool | None): Keyword arguments.
        """
        self._generator = generator
        super().__init__(*args, **kwargs)

    @property
    def generator(self) -> FallbackPrefixProtocol | CommonASTVisitor:
        """Get the delegate generator.

        Returns:
            FallbackPrefixProtocol | CommonASTVisitor: The generator or self.
        """
        return getattr(self, "_generator", None) or self

    def get_fallback_prefix(self) -> str:
        """Return the backend prefix (e.g., 'jax', 'pt', 'mx').

        Returns:
            str: Result.
        """
        return ""
