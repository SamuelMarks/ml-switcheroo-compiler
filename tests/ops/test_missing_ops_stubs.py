"""Test missing_ops_stubs.py."""

from ml_switcheroo_compiler.ops.generated import missing_ops_stubs


def test_missing_ops_stubs():
    """Test missing_ops_stubs."""
    assert missing_ops_stubs.__all__ == []
