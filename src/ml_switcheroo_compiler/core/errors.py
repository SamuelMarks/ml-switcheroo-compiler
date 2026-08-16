# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module errors.py."""

import os
from typing import Any

import yaml

"""Error classes for the ml-switcheroo compiler."""

_ERROR_TEMPLATES: dict[str, str] = {}


def _load_error_templates() -> None:
    """_load_error_templates function.

    Returns:
        Any: Result.
    """
    global _ERROR_TEMPLATES
    if not _ERROR_TEMPLATES:
        yaml_path = os.path.join(os.path.dirname(__file__), "error_messages.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                _ERROR_TEMPLATES = yaml.safe_load(f) or {}


class SwitcherooError(Exception):
    """Define base class for all ml-switcheroo errors."""

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        """__init__ function.

        Args:
            message: The message.
            kwargs: Additional kwargs.

        Args:
            input_vars (list): The input vars.
            node (Any): The node.
            **kwargs (Any): Keyword arguments.
        self (Any): The self parameter.
        message (Any): The message parameter.

        Returns:
        Any: Result.
        """
        _load_error_templates()
        template = _ERROR_TEMPLATES.get(self.__class__.__name__)
        if template and kwargs:
            try:
                message = template.format(**kwargs)
            except KeyError:
                pass
        super().__init__(message)


class TracingError(SwitcherooError):
    """Raised when control flow breaks proxy tensor tracing."""


class CompilationError(SwitcherooError):
    """Raised during IR generation or pass failure."""


class ShapeMismatchError(SwitcherooError):
    """Raised during static shape inference."""


class DTypePromotionError(SwitcherooError):
    """Raised on invalid automatic type casting."""


class BackendNotSupportedError(SwitcherooError):
    """Raised when an edge target lacks an op."""


class UnimplementedMathError(SwitcherooError):
    """Raised when an operation lacks a mathematical equivalent."""


class MissingJVPRuleError(SwitcherooError):
    """Exception raised when a required JVP/VJP rule is missing for an operation."""
