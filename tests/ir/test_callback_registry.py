"""Unit tests for `ml_switcheroo_compiler.ir.callback_registry`."""

from __future__ import annotations

import pytest
from ml_switcheroo_ir import CustomRuleHandleProtocol

from ml_switcheroo_compiler.ir.callback_registry import (
    CustomRuleHandle,
    clear_custom_rule_handles,
    register_custom_rule_handle,
    resolve_custom_rule_handle,
)


def test_register_and_invoke_custom_rule_handle() -> None:
    """Verify registration, protocol conformance, idempotency, invocation, and error branches."""
    assert register_custom_rule_handle(None) is None

    def _double(x: int) -> int:
        """Double input integer.

        Args:
            x (int): Input scalar.

        Returns:
            int: Doubled scalar.
        """
        return x * 2

    handle = register_custom_rule_handle(_double, kind="vjp")
    assert isinstance(handle, CustomRuleHandle)
    assert isinstance(handle, CustomRuleHandleProtocol)
    assert handle.handle_id == handle.rule_id
    assert handle.kind == "vjp"
    assert register_custom_rule_handle(handle) is handle
    assert handle.resolve() is _double
    assert resolve_custom_rule_handle(handle) is _double
    assert handle(21) == 42

    with pytest.raises(KeyError):
        CustomRuleHandle(rule_id="nonexistent_rule").resolve()

    import ml_switcheroo_compiler.ir.callback_registry as cb_mod

    saved_callbacks = dict(cb_mod._CALLBACK_TABLE)
    try:
        clear_custom_rule_handles()
        with pytest.raises(KeyError):
            resolve_custom_rule_handle(handle)
    finally:
        cb_mod._CALLBACK_TABLE.update(saved_callbacks)
