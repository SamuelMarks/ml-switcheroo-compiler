# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Error classes for the ml-switcheroo compiler."""


class SwitcherooError(Exception):
    """Define base class for all ml-switcheroo errors."""


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
