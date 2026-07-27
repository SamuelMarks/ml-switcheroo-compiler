# ruff: noqa: D103
"""Tests for serialization extras."""

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


def test_serialization_extras() -> None:
    assert TrackableResource() is not None
    assert PythonState() is not None

    p1 = MaxShardSizePolicy(100)
    assert p1.max_shard_size == 100

    assert ShardByTaskPolicy() is not None

    sm = SavedModel()
    sm.save("path")
    assert isinstance(SavedModel.load("path"), SavedModel)

    assert read_fingerprint("path") == "fingerprint"
    assert load_variable("path", "name") is not None
    run_restore_ops("path")
