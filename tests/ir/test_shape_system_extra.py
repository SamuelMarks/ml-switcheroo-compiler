"""Test shape system extra."""

from ml_switcheroo_compiler.ir.shape_system import SymInt


def test_symint_eq() -> None:
    """Test symint eq branch."""
    s1 = SymInt("A")
    assert not (s1 == "A")
