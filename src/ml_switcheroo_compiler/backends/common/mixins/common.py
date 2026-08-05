# ruff: noqa: E501
"""Provide mixin module."""

from __future__ import annotations


class CommonASTVisitor:
    """Define base mixin/visitor for shared AST generation logic across backends."""

    def __init__(self, *args: object, generator: object = None, **kwargs: object) -> None:
        """Initialize the visitor.

        Args:
            generator (object): The generator.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Args:
            generator (object): The generator.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.
        """
        self._generator = generator
        super().__init__(*args, **kwargs)

    @property
    def generator(self) -> object:
        """Get the delegate generator.

        Returns:
            object: The generator or self.
        """
        return getattr(self, "_generator", None) or self

    def _get_backend_prefix(self) -> str:
        """Return the backend prefix (e.g., 'jax', 'pt', 'mx').

        Returns:
        str: Result.
        """
        return ""
