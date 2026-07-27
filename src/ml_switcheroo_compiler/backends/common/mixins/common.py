# ruff: noqa: E501
"""Mixin module."""

from __future__ import annotations


class CommonASTVisitor:
    """Base mixin/visitor for shared AST generation logic across backends."""

    def __init__(self, *args: object, generator: object = None, **kwargs: object) -> None:
        """Initialize the visitor.

        Args:
            *args (object): Additional positional arguments.
            generator (object, optional): The generator to delegate to. Defaults to None.
            **kwargs (object): Additional keyword arguments.
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
        """Returns the backend prefix (e.g., 'jax', 'pt', 'mx')."""
        return ""
