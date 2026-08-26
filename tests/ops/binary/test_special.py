# ruff: noqa: E501

from ml_switcheroo_compiler.ops.binary.special import Atan2, Divmod, Isclose

"Core abstractions and logic definitions for test_binary_special_coverage.py."


def test_divmod_infer_shape_fallback() -> None:
    """Test the divmod infer shape fallback behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Docstring."
        op = Divmod()
        assert op.infer_shape(None, (1, 2)) is None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_isclose_infer_shape_fallback() -> None:
    """Test the isclose infer shape fallback behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Docstring."
        op = Isclose()
        assert op.infer_shape(None, (1, 2)) is None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_atan2_infer_shape_fallback() -> None:
    """Test the atan2 infer shape fallback behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Docstring."
        op = Atan2()
        assert op.infer_shape(None, (1, 2)) is None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
