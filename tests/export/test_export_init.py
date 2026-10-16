"""Test module."""

from __future__ import annotations

from ml_switcheroo_compiler.export import ExportArchive


def test_export_init() -> None:
    """Test function test_export_init."""
    assert ExportArchive is not None
