"""Comprehensive tests for serialization, model loading, and checkpoint restoration."""

from __future__ import annotations

import os
import tempfile
import zipfile
from typing import Any

import numpy as np
import pytest

from ml_switcheroo_compiler.serialization import (
    LoadedModel,
    SavedModel,
    custom_object_scope,
    deserialize_keras_object,
    get_registered_object,
    graph_to_json,
    load_model,
    read_fingerprint,
    register_keras_serializable,
    run_restore_ops,
    save_model,
    serialize_keras_object,
)


def test_serialization_primitives_and_custom_objects() -> None:
    """Test custom object registration, scoping, and recursive configuration serialization."""
    assert deserialize_keras_object() == {}
    assert serialize_keras_object() == {}

    # Custom serializable class
    @register_keras_serializable(package="MyPackage", name="CustomDense")
    class CustomDense:
        def __init__(self, units: int = 32) -> None:
            self.units = units

        def get_config(self) -> dict[str, Any]:
            return {"units": self.units}

        @classmethod
        def from_config(cls, cfg: dict[str, Any]) -> CustomDense:
            return cls(**cfg)

    obj = CustomDense(units=64)
    serialized = serialize_keras_object(obj)
    assert serialized["class_name"] == "CustomDense"
    assert serialized["config"]["units"] == 64

    # Deserialization from registry
    deserialized = deserialize_keras_object(serialized)
    assert isinstance(deserialized, CustomDense)
    assert deserialized.units == 64

    # Scoped custom object
    class ScopedLayer:
        def __init__(self, rate: float = 0.5) -> None:
            self.rate = rate

        def get_config(self) -> dict[str, Any]:
            return {"rate": self.rate}

    with custom_object_scope({"ScopedLayer": ScopedLayer}):
        assert get_registered_object("ScopedLayer") is ScopedLayer
        s_cfg = {"class_name": "ScopedLayer", "config": {"rate": 0.25}}
        des_scoped = deserialize_keras_object(s_cfg)
        assert isinstance(des_scoped, ScopedLayer)
        assert des_scoped.rate == 0.25

    # Scope restoration
    assert get_registered_object("ScopedLayer") is None

    # Unknown class raises ValueError
    with pytest.raises(ValueError, match="Unknown class 'UnregisteredClass'"):
        deserialize_keras_object({"class_name": "UnregisteredClass", "config": {}})


def test_universal_model_loader_and_save_model() -> None:
    """Test save_model and load_model roundtrip, metadata extraction, and corruption detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.keras")

        # Mock model with config and weights
        class DummyModel:
            def __init__(self) -> None:
                self.weights = [np.array([1.0, 2.0], dtype=np.float32)]

            def get_config(self) -> dict[str, Any]:
                return {"architecture": "linear", "hidden_dim": 128}

        m = DummyModel()
        save_model(m, model_path)
        assert os.path.exists(model_path)

        # Successful load
        loaded = load_model(model_path)
        assert isinstance(loaded, LoadedModel)
        assert loaded.config["architecture"] == "linear"
        assert loaded.config["hidden_dim"] == 128
        assert "keras_version" in loaded.metadata

        # Error cases: non-existent file
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model(os.path.join(tmpdir, "missing.keras"))

        # Error case: directory passed instead of zip archive
        with pytest.raises(ValueError, match="is a directory"):
            load_model(tmpdir)

        # Error case: damaged / invalid zip
        damaged_zip = os.path.join(tmpdir, "corrupt.keras")
        with open(damaged_zip, "wb") as f:
            f.write(b"Not a zip file content")
        with pytest.raises(ValueError, match="Invalid or corrupted model archive"):
            load_model(damaged_zip)

        # Error case: zip missing config.json
        incomplete_zip = os.path.join(tmpdir, "incomplete.keras")
        with zipfile.ZipFile(incomplete_zip, "w") as zf:
            zf.writestr("metadata.json", "{}")
        with pytest.raises(ValueError, match="missing 'config.json'"):
            load_model(incomplete_zip)


def test_saved_model_and_fingerprinting() -> None:
    """Test SavedModel save/load cycles, proto manifest generation, and deterministic fingerprinting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm_dir = os.path.join(tmpdir, "my_saved_model")

        class MockGraphModel:
            def get_config(self) -> dict[str, Any]:
                return {"op": "Add", "inputs": ["x", "y"]}

        sm = SavedModel(model=MockGraphModel(), signatures={"serving_default": {"inputs": ["x"]}})
        sm.save(sm_dir)

        assert os.path.exists(os.path.join(sm_dir, "saved_model.pb"))
        assert os.path.exists(os.path.join(sm_dir, "fingerprint.pb"))

        # Read fingerprint
        fp = read_fingerprint(sm_dir)
        assert len(fp) == 64  # SHA-256 hex digest length

        # Fingerprint invariance for identical structure
        sm_dir2 = os.path.join(tmpdir, "my_saved_model2")
        sm2 = SavedModel(model=MockGraphModel(), signatures={"serving_default": {"inputs": ["x"]}})
        sm2.save(sm_dir2)
        fp2 = read_fingerprint(sm_dir2)
        assert fp == fp2

        # Fingerprint changes on mutation
        sm_mutated_dir = os.path.join(tmpdir, "mutated_model")
        sm_mut = SavedModel(model=MockGraphModel(), signatures={"serving_default": {"inputs": ["x", "z"]}})
        sm_mut.save(sm_mutated_dir)
        fp_mut = read_fingerprint(sm_mutated_dir)
        assert fp != fp_mut

        # SavedModel load
        loaded_sm = SavedModel.load(sm_dir)
        assert loaded_sm.signatures == {"serving_default": {"inputs": ["x"]}}
        assert loaded_sm.model["op"] == "Add"

        # Corrupted proto load
        corrupt_pb_dir = os.path.join(tmpdir, "corrupt_sm")
        os.makedirs(corrupt_pb_dir, exist_ok=True)
        with open(os.path.join(corrupt_pb_dir, "saved_model.pb"), "wb") as f:
            f.write(b"{corrupted invalid json]")
        with pytest.raises(ValueError, match="Corrupted SavedModel proto"):
            SavedModel.load(corrupt_pb_dir)


def test_variable_restoration_engine() -> None:
    """Test checkpoint parsing across NPZ, NPY, shape validation, and mismatch errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Non-existent checkpoint path
        with pytest.raises(FileNotFoundError, match="not found"):
            run_restore_ops(os.path.join(tmpdir, "does_not_exist"))

        # Create checkpoint directory with NPY and NPZ variables
        var1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        var2 = np.array([10.0, 20.0], dtype=np.float32)

        np.save(os.path.join(tmpdir, "dense_w.npy"), var1)
        np.savez(os.path.join(tmpdir, "weights.npz"), dense_b=var2)

        # Restore without target model
        restored = run_restore_ops(tmpdir)
        assert "dense_w" in restored
        assert "dense_b" in restored
        assert np.allclose(restored["dense_w"], var1)
        assert np.allclose(restored["dense_b"], var2)

        # Restore into target model with matching shapes
        class MockVar:
            def __init__(self, name: str, shape: tuple[int, ...]) -> None:
                self.name = name
                self.shape = shape
                self.assigned = None

            def assign(self, val: np.ndarray) -> None:
                self.assigned = val

        class TargetModel:
            def __init__(self) -> None:
                self.weights = [
                    MockVar("dense_w", (2, 2)),
                    MockVar("dense_b", (2,)),
                ]

        model_inst = TargetModel()
        run_restore_ops(tmpdir, target_model=model_inst)
        assert np.allclose(model_inst.weights[0].assigned, var1)
        assert np.allclose(model_inst.weights[1].assigned, var2)

        # Shape mismatch in target model raises ValueError
        class IncompatibleModel:
            def __init__(self) -> None:
                self.weights = [MockVar("dense_w", (3, 3))]

        with pytest.raises(ValueError, match="Shape mismatch restoring variable 'dense_w'"):
            run_restore_ops(tmpdir, target_model=IncompatibleModel())

        # Shape mismatch in variables dictionary raises ValueError
        with pytest.raises(ValueError, match="Shape mismatch restoring variable 'dense_b'"):
            run_restore_ops(tmpdir, variables={"dense_b": MockVar("dense_b", (5,))})


def test_file_handler() -> None:
    """Test file handler."""

    class DummyGraph:
        def to_json(self):
            return "{}"

    assert graph_to_json(DummyGraph()) == "{}"
