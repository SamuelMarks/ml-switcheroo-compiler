import pytest  # noqa: F401, I001
from unittest.mock import patch, MagicMock  # noqa: F401
from ml_switcheroo_compiler.backends.snapshot_grounding import SnapshotGroundingEngine


def test_grounding_engine_broken_import():
    """Test grounding engine gracefully handles importlib failures."""
    with patch("importlib.import_module", side_effect=ImportError("broken import")):
        engine = SnapshotGroundingEngine("/dummy/dir")
        assert engine._grounding_engine is None


def test_grounding_engine_unreadable_dir(monkeypatch):
    """Test unreadable snapshot directory handling."""
    with patch("os.path.isdir", return_value=True), patch("os.listdir", side_effect=PermissionError("unreadable")):
        engine = SnapshotGroundingEngine("/dummy/dir")
        assert len(engine._raw_snapshots) == 0
