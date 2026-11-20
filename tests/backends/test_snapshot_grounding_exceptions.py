"""Tests for test_snapshot_grounding_exceptions."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch  # noqa: F401

import pytest

from ml_switcheroo_compiler.backends.snapshot_grounding import SnapshotGroundingEngine

if typing.TYPE_CHECKING:
    pass


def test_grounding_engine_broken_import() -> None:
    """Test grounding engine gracefully handles importlib failures."""
    with patch("importlib.import_module", side_effect=ImportError("broken import")):
        engine = SnapshotGroundingEngine("/dummy/dir")
        assert engine._grounding_engine is None


def test_grounding_engine_unreadable_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test unreadable snapshot directory handling.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    with patch("os.path.isdir", return_value=True), patch("os.listdir", side_effect=PermissionError("unreadable")):
        engine = SnapshotGroundingEngine("/dummy/dir")
        assert len(engine._raw_snapshots) == 0
