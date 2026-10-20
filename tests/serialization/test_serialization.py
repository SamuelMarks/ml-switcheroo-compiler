"""Comprehensive tests for serialization, model loading, and checkpoint restoration."""

from __future__ import annotations

import os
import tempfile
import typing
import zipfile

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.serialization import (
    LoadedModel,
    SavedModel,
    custom_object_scope,
    deserialize_object,
    get_registered_object,
    graph_to_json,
    load_model,
    read_fingerprint,
    register_serializable,
    run_restore_ops,
    save_model,
    serialize_object,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_serialization_primitives_and_custom_objects() -> None:
    """Test custom object registration, scoping, and recursive configuration serialization."""
    assert deserialize_object() == {}
    assert serialize_object() == {}

    # Custom serializable class
    @register_serializable(package="MyPackage", name="CustomDense")
    class CustomDense:
        """Test class CustomDense."""

        def __init__(self, units: int = 32) -> None:
            """Test function __init__.

            Args:
                units: Test parameter `units`.
            """
            self.units = units

        def get_config(self) -> dict[str, EagerValue]:
            """Test function get_config."""
            return {"units": self.units}

        @classmethod
        def from_config(cls, cfg: dict[str, EagerValue]) -> CustomDense:
            """Test function from_config.

            Args:
                cfg: Test parameter `cfg`.
            """
            return typing.cast("CustomDense", typing.cast("EagerCallableProtocol", cls)(**cfg))

    obj = CustomDense(units=64)
    serialized = serialize_object(typing.cast("EagerValue", obj))
    assert typing.cast("EagerTensorProtocol", getattr(serialized, "__getitem__")("class_name")) == "CustomDense"
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(serialized, "__getitem__")("config")), "__getitem__")("units")) == 64

    # Deserialization from registry
    deserialized = typing.cast("CustomDense", deserialize_object(serialized))
    assert isinstance(deserialized, CustomDense)
    assert deserialized.units == 64

    # Scoped custom object
    class ScopedLayer:
        """Test class ScopedLayer."""

        def __init__(self, rate: float = 0.5) -> None:
            """Test function __init__.

            Args:
                rate: Test parameter `rate`.
            """
            self.rate = rate

        def get_config(self) -> dict[str, EagerValue]:
            """Test function get_config."""
            return {"rate": self.rate}

    with custom_object_scope({"ScopedLayer": ScopedLayer}):
        assert get_registered_object("ScopedLayer") is ScopedLayer
        s_cfg = typing.cast("EagerValue", {"class_name": "ScopedLayer", "config": {"rate": 0.25}})
        des_scoped = typing.cast("ScopedLayer", deserialize_object(s_cfg))
        assert isinstance(des_scoped, ScopedLayer)
        assert des_scoped.rate == 0.25

    # Scope restoration
    assert get_registered_object("ScopedLayer") is None

    # Unknown class raises ValueError
    with pytest.raises(ValueError, match="Unknown class 'UnregisteredClass'"):
        deserialize_object(typing.cast("EagerValue", {"class_name": "UnregisteredClass", "config": {}}))


def test_universal_model_loader_and_save_model() -> None:
    """Test save_model and load_model roundtrip, metadata extraction, and corruption detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.keras")

        # Mock model with config and weights
        class DummyModel:
            """Test class DummyModel."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.weights = [np.array([1.0, 2.0], dtype=np.float32)]

            def get_config(self) -> dict[str, EagerValue]:
                """Test function get_config."""
                return {"architecture": "linear", "hidden_dim": 128}

        m = DummyModel()
        typing.cast("EagerCallableProtocol", save_model)(m, model_path)
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
            """Test class MockGraphModel."""

            def get_config(self) -> dict[str, EagerValue]:
                """Test function get_config."""
                return {"op": "Add", "inputs": ["x", "y"]}

        sm = SavedModel(model=typing.cast("EagerValue", MockGraphModel()), signatures={"serving_default": {"inputs": ["x"]}})
        sm.save(sm_dir)

        assert os.path.exists(os.path.join(sm_dir, "saved_model.pb"))
        assert os.path.exists(os.path.join(sm_dir, "fingerprint.pb"))

        # Read fingerprint
        fp = read_fingerprint(sm_dir)
        assert len(fp) == 64  # SHA-256 hex digest length

        # Fingerprint invariance for identical structure
        sm_dir2 = os.path.join(tmpdir, "my_saved_model2")
        sm2 = SavedModel(model=typing.cast("EagerValue", MockGraphModel()), signatures={"serving_default": {"inputs": ["x"]}})
        sm2.save(sm_dir2)
        fp2 = read_fingerprint(sm_dir2)
        assert fp == fp2

        # Fingerprint changes on mutation
        sm_mutated_dir = os.path.join(tmpdir, "mutated_model")
        sm_mut = SavedModel(model=typing.cast("EagerValue", MockGraphModel()), signatures={"serving_default": {"inputs": ["x", "z"]}})
        sm_mut.save(sm_mutated_dir)
        fp_mut = read_fingerprint(sm_mutated_dir)
        assert fp != fp_mut

        # SavedModel load
        loaded_sm = SavedModel.load(sm_dir)
        assert loaded_sm.signatures == {"serving_default": {"inputs": ["x"]}}
        assert typing.cast("EagerTensorProtocol", getattr(loaded_sm.model, "__getitem__")("op")) == "Add"

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
        assert typing.cast("EagerCallableProtocol", np.allclose)(restored["dense_w"], var1)
        assert typing.cast("EagerCallableProtocol", np.allclose)(restored["dense_b"], var2)

        # Restore into target model with matching shapes
        class MockVar:
            """Test class MockVar."""

            def __init__(self, name: str, shape: tuple[int, ...]) -> None:
                """Test function __init__.

                Args:
                    name: Test parameter `name`.
                    shape: Test parameter `shape`.
                """
                self.name = name
                self.shape = shape
                self.assigned = None

            def assign(self, val: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
                """Test function assign.

                Args:
                    val: Test parameter `val`.
                """
                setattr(self, "assigned", val)

        class TargetModel:
            """Test class TargetModel."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.weights = [
                    MockVar("dense_w", (2, 2)),
                    MockVar("dense_b", (2,)),
                ]

        model_inst = TargetModel()
        run_restore_ops(tmpdir, target_model=typing.cast("EagerValue", model_inst))
        assert typing.cast("EagerCallableProtocol", np.allclose)(model_inst.weights[0].assigned, var1)
        assert typing.cast("EagerCallableProtocol", np.allclose)(model_inst.weights[1].assigned, var2)

        # Shape mismatch in target model raises ValueError
        class IncompatibleModel:
            """Test class IncompatibleModel."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.weights = [MockVar("dense_w", (3, 3))]

        with pytest.raises(ValueError, match="Shape mismatch restoring variable 'dense_w'"):
            run_restore_ops(tmpdir, target_model=typing.cast("EagerValue", IncompatibleModel()))

        # Shape mismatch in variables dictionary raises ValueError
        with pytest.raises(ValueError, match="Shape mismatch restoring variable 'dense_b'"):
            run_restore_ops(tmpdir, variables={"dense_b": MockVar("dense_b", (5,))})


def test_file_handler() -> None:
    """Test file handler."""

    class DummyGraph:
        """Test class DummyGraph."""

        def to_json(self) -> EagerValue:
            """Test function to_json."""
            return "{}"

    assert typing.cast("EagerCallableProtocol", graph_to_json)(typing.cast("EagerCallableProtocol", DummyGraph)()) == "{}"
