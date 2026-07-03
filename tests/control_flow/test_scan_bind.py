"""Module docstring."""

from ml_switcheroo_compiler.ops.control_flow import scan_bind


def test_scan_bind_coverage() -> object:
    """Function docstring."""
    res = scan_bind(lambda x: x, [1, 2, 3])
    assert res is not None
