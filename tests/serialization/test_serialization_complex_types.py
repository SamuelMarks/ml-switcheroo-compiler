# ruff: noqa: D103
"""Tests for serialization extras."""

from pathlib import Path

from ml_switcheroo_compiler.serialization import (
    MaxShardSizePolicy,
    PythonState,
    SavedModel,
    ShardByTaskPolicy,
    TrackableResource,
    load_variable,
    read_fingerprint,
    run_restore_ops,
)


def test_serialization_extras(tmp_path: Path) -> None:
    assert TrackableResource() is not None
    assert PythonState() is not None

    p1 = MaxShardSizePolicy(100)
    assert p1.max_shard_size == 100

    assert ShardByTaskPolicy() is not None

    save_dir: str = str(tmp_path / "saved_model")
    sm = SavedModel()
    sm.save(save_dir)
    assert isinstance(SavedModel.load(save_dir), SavedModel)

    fp = read_fingerprint(save_dir)
    assert len(fp) == 64
    assert load_variable(save_dir, "name") is not None
    run_restore_ops(save_dir)
