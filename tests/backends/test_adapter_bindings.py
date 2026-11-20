"""Unit tests verifying 100% branch coverage of static structural adapter bindings."""

from __future__ import annotations

from unittest.mock import MagicMock

from ml_switcheroo_compiler.backends.adapter_bindings import ADAPTER_ATTR_READERS, read_static_attr
from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class _FailingAttrTarget:
    """Target raising `AttributeError` dynamically on second property access after structural check."""

    def __init__(self) -> None:
        """Initialize access counter."""
        self._reads = 0

    @property
    def abs(self) -> int:
        """Return `1` on protocol `isinstance` probe, then raise `AttributeError` on value read.

        Returns:
            int: Sentinel value during structural protocol check.

        Raises:
            AttributeError: Raised on subsequent value read to exercise `read_static_attr` exception handling.
        """
        self._reads += 1
        if self._reads > 1:
            raise AttributeError("absent")
        return 1


class _EmptyTarget:
    """Minimal object with zero custom attributes for absent-branch verification."""


def test_adapter_attr_readers_all_branches() -> None:
    """Verify every static attribute reader exercises both present and absent branches."""
    present_mock = MagicMock(unsafe=True)
    absent_target = _EmptyTarget()

    assert len(ADAPTER_ATTR_READERS) > 100
    for name, reader in ADAPTER_ATTR_READERS.items():
        found_yes, _ = reader(present_mock)
        assert found_yes is True, f"Expected present branch for {name}"

        found_no, val_no = reader(absent_target)
        assert found_no is False, f"Expected absent branch for {name}"
        assert val_no is None


def test_read_static_attr_dispatch_and_error_branches() -> None:
    """Verify `read_static_attr` handles unknown names, present values, and `AttributeError`."""
    assert read_static_attr(MagicMock(), "__nonexistent_symbol_xyz__") == (False, None)

    present = MagicMock()
    present.abs = 123
    res_present: tuple[bool, BackendArg] = read_static_attr(present, "abs")
    assert res_present == (True, 123)
    assert read_static_attr(_FailingAttrTarget(), "abs") == (False, None)
