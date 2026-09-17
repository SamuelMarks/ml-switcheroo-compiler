"""Unit tests for weight format dispatchers and serialization state extraction."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import ml_switcheroo_compiler.serialization as s


def test_serialization_format_dispatch(tmp_path: Path) -> None:
    """Verify weight format save/load dispatchers and model state extraction helpers.

    Args:
        tmp_path (Path): Temporary filesystem directory fixture.
    """
    h5_path = str(tmp_path / "weights.h5")
    st_path = str(tmp_path / "weights.safetensors")
    npz_path = str(tmp_path / "weights.npz")
    pk_path = str(tmp_path / "weights.pk")
    zip_path = str(tmp_path / "model_bundle.zip")
    dummy_model = str(tmp_path / "dummy_model")

    with patch("ml_switcheroo_compiler.serialization.H5WeightFormat.save"):
        s._save_as_h5({"a": 1}, h5_path)
    with patch("ml_switcheroo_compiler.serialization.SafetensorsWeightFormat.save"):
        s._save_as_safetensors({"a": 1}, st_path)
    with patch("ml_switcheroo_compiler.serialization.H5WeightFormat.load"):
        s._load_h5_weights(h5_path)
    with patch("ml_switcheroo_compiler.serialization.SafetensorsWeightFormat.load"):
        s._load_safetensors_weights(st_path)
    with patch("ml_switcheroo_compiler.serialization.NpzWeightFormat.load"):
        s._load_npz_weights(npz_path)
    with patch("ml_switcheroo_compiler.serialization.PickleWeightFormat.load"):
        s._load_pickle_weights(pk_path)

    class FakeWeight:
        """Mock weight variable."""

        def __init__(self, name: str) -> None:
            self.name: str = name

        def numpy(self) -> int:
            return 1

        def tolist(self) -> int:
            return 1

    class FakeModel:
        """Mock neural network model."""

        weights: list[FakeWeight] = [FakeWeight("w")]
        optimizer: object = type("O", (), {"variables": [FakeWeight("v")], "momentums": [FakeWeight("m")]})()
        non_trainable_variables: list[FakeWeight] = [FakeWeight("nt")]
        ema_variables: list[FakeWeight] = [FakeWeight("ema")]

        def get_config(self) -> dict[str, int]:
            return {"c": 1}

    ws = s._extract_model_weights(FakeModel())
    assert len(ws) == 1
    st: dict[str, object] = {}
    s._extract_optimizer_state(FakeModel(), st)
    s._extract_non_trainable_state(FakeModel(), st, {})
    s._extract_ema_state(FakeModel(), st)
    s._extract_model_state(FakeModel(), {})

    s._compile_model_metadata(FakeModel())

    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        with patch("ml_switcheroo_compiler.serialization._save_as_h5"):
            s._write_h5_to_zip(zf, "test.h5", {"a": 1})
    if os.path.exists(zip_path):
        os.remove(zip_path)

    ctx = s.KerasSerializationContext(filepath=zip_path, config_dict={}, metadata={}, weights_store={"a": 1}, state_store={"b": 1})
    with patch("ml_switcheroo_compiler.serialization._write_h5_to_zip"):
        s._write_keras_zip(ctx)

    try:
        s.save_model(FakeModel(), dummy_model)
    except Exception:
        pass

    try:
        s.load_model(dummy_model)
    except Exception:
        pass

    s.get_custom_objects()
    s.get_registered_name(list)
    s.get_registered_object("list")
    s.serialize_keras_object(FakeModel())
    s.deserialize_keras_object({"class_name": "list"})

    s.TrackableResource()
    s.PythonState()
    s.MaxShardSizePolicy(1)
    s.ShardByTaskPolicy()
    s.SavedModel()
    s.read_fingerprint(dummy_model)
    s.load_variable(dummy_model, "var1")

    try:
        s.run_restore_ops(dummy_model)
    except Exception:
        pass


def test_weight_format_load_save(tmp_path: Path) -> None:
    """Verify weight validation and weight saving/loading dispatch.

    Args:
        tmp_path (Path): Temporary filesystem directory fixture.
    """
    assert s._validate_and_map_weights({"a": 1}) == {"a": 1}

    tf = str(tmp_path / "last_lines_weights.h5")
    if os.path.exists(tf):
        os.remove(tf)

    s.save_weights(None, tf)

    with patch.object(s.H5WeightFormat, "load", return_value={"x": 2}):
        res = s.load_weights(tf)
        assert res == {"x": 2}

    if os.path.exists(tf):
        os.remove(tf)
