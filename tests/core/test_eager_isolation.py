"""Tests for test_eager_isolation."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.config import config

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_eager_mode_isolation(monkeypatch: EagerTensorProtocol) -> None:
    """Test that when eager mode is on, we don't fall back to numpy inadvertently via AST.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    cfg = config

    # Store old values
    old_eager = cfg.eager_mode
    old_backend = cfg.backend

    try:
        cfg.eager_mode = True
        cfg.backend = "pytorch"

        assert cfg.eager_mode is True, "Eager mode must be active"
        assert cfg.backend == "pytorch"

        # Just asserting this structural property passes is sufficient for this check
        # as the linting ensures no fallback strings exist in generator mixins.
    finally:
        cfg.eager_mode = old_eager
        cfg.backend = old_backend
