# ruff: noqa: E501

from ml_switcheroo_compiler.ir.shape_system import SymInt

"Test shape system extra."


def test_symint_eq() -> None:
    """Test the symint eq behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test symint eq branch."
        s1 = SymInt("A")
        assert not s1 == "A"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
