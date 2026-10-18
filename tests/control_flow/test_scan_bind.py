# ruff: noqa: E501
"""Core abstractions and logic definitions for test_scan_bind.py."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.control_flow import scan_bind


def test_scan_bind_coverage() -> None:
    """Test the scan bind coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    res = scan_bind(lambda x: x, [1, 2, 3])
    assert res is not None
