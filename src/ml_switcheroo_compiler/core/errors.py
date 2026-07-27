"""Error classes for the ml-switcheroo compiler."""


class SwitcherooError(Exception):
    """Base class for all ml-switcheroo errors."""


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
