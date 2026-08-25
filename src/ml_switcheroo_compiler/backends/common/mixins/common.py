"""Module common.py."""

from __future__ import annotations

from typing import Any, Optional

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""


class CommonASTVisitor:
    """Define base mixin/visitor for shared AST generation logic across backends."""

    def __init__(self, *args: Any, generator: Any | None = None, **kwargs: Any) -> None:
        """Initialize the visitor.

        Args:
            generator (Any): The generator.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.
        """
        self._generator = generator
        super().__init__(*args, **kwargs)

    @property
    def generator(self) -> Any:
        """Get the delegate generator.

        Returns: Any: The generator or self.
        """
        return getattr(self, "_generator", None) or self

    def get_fallback_prefix(self) -> str:
        """Return the backend prefix (e.g., 'jax', 'pt', 'mx').

        Returns:
        str: Result.
        """
        return ""
