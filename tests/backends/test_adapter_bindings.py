"""Unit tests verifying 100% branch coverage of static structural adapter bindings."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

from ml_switcheroo_compiler.backends.adapter_bindings import ADAPTER_ATTR_READERS, read_static_attr
from ml_switcheroo_compiler.backends.eager_registry import BackendArg


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
    assert read_static_attr(types.ModuleType("empty_mod"), "abs") == (False, None)
    assert read_static_attr(MagicMock(spec=[]), "abs") == (False, None)

    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs
    from ml_switcheroo_compiler.core.shape import extract_data
    from tests.testing_utils import MockTensorStub

    deleted_mock = MagicMock()
    _ = deleted_mock.abs
    del deleted_mock.abs
    assert is_mock_with_attrs(deleted_mock, "abs") is False
    stub_no_data = MockTensorStub()
    assert extract_data(stub_no_data) is stub_no_data
