"""Error classes for the ml-switcheroo compiler."""


class SwitcherooError(Exception):
    """Base class for all ml-switcheroo errors."""

    pass


class TracingError(SwitcherooError):
    """Raised when control flow breaks proxy tensor tracing."""

    pass


class CompilationError(SwitcherooError):
    """Raised during IR generation or pass failure."""

    pass


class ShapeMismatchError(SwitcherooError):
    """Raised during static shape inference."""

    pass


class DTypePromotionError(SwitcherooError):
    """Raised on invalid automatic type casting."""

    pass


class BackendNotSupportedError(SwitcherooError):
    """Raised when an edge target lacks an op."""

    pass


class UnimplementedMathError(SwitcherooError):
    """Raised if NumPy/SciPy fallback is missing in Eager Mode."""

    pass
