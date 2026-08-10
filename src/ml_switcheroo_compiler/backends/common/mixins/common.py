from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""
from typing import Any


class CommonASTVisitor:
    """Define base mixin/visitor for shared AST generation logic across backends."""

    def __init__(self, *args: Any, generator: Any = None, **kwargs: Any) -> None:
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
    def generator(self) -> Any:
        """Get the delegate generator.

        Returns: Any: The generator or self.
        """
        return getattr(self, "_generator", None) or self

    def _get_backend_prefix(self) -> str:
        """Return the backend prefix (e.g., 'jax', 'pt', 'mx').

        Returns:
        str: Result.
        """
        return ""
