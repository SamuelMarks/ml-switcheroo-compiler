"""Comprehensive unit tests covering all edge cases and branches in ml_switcheroo_compiler.serialization."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.serialization import (
    SavedModel,
    _serialize_nested_config,
    deserialize_keras_object,
    get_registered_name,
    load_model,
    read_fingerprint,
    register_keras_serializable,
    run_restore_ops,
    serialize_keras_object,
)


def test_load_model_edge_branches() -> None:
    """Test branches in load_model: metadata.json, invalid zip / missing files, exception handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Archive with metadata.json and config.json
        model_path = os.path.join(tmpdir, "test.keras")
        with zipfile.ZipFile(model_path, "w") as zf:
            zf.writestr("config.json", json.dumps({"name": "test_model"}))
            zf.writestr("metadata.json", json.dumps({"author": "switcheroo"}))

        # 0. Archive with model.weights.h5
        h5_model_path = os.path.join(tmpdir, "h5_model.keras")
        with zipfile.ZipFile(h5_model_path, "w") as zf:
            zf.writestr("config.json", json.dumps({"name": "h5_model"}))
            zf.writestr("model.weights.h5", b"")

        # Test case 1: tmp_path exists and is removed
        with patch("ml_switcheroo_compiler.serialization._load_h5_weights", return_value={"w": 1}):
            loaded_h5 = load_model(h5_model_path)
            assert loaded_h5.weights == {"w": 1}

        # Test case 2: tmp_path does not exist when entering finally (e.g. removed beforehand)
        def remove_and_load(p):
            if os.path.exists(p):
                os.remove(p)
            return {"w": 2}

        with patch("ml_switcheroo_compiler.serialization._load_h5_weights", side_effect=remove_and_load):
            loaded_h5_2 = load_model(h5_model_path)
            assert loaded_h5_2.weights == {"w": 2}

        loaded = load_model(model_path)
        assert loaded.config == {"name": "test_model"}
        assert loaded.metadata == {"author": "switcheroo"}
        assert loaded.weights == {}

        # 2. Archive without metadata.json (triggers branch where metadata.json is not present)
        no_meta_path = os.path.join(tmpdir, "no_meta.keras")
        with zipfile.ZipFile(no_meta_path, "w") as zf:
            zf.writestr("config.json", json.dumps({"name": "no_meta"}))

        loaded_no_meta = load_model(no_meta_path)
        assert loaded_no_meta.metadata == {}

        # 3. Zip missing config.json -> ValueError directly
        missing_cfg_path = os.path.join(tmpdir, "missing_cfg.keras")
        with zipfile.ZipFile(missing_cfg_path, "w") as zf:
            zf.writestr("other.txt", "data")

        with pytest.raises(ValueError, match="is missing 'config.json'"):
            load_model(missing_cfg_path)

        # 4. Valid zip raising a non-ValueError/FileNotFoundError inside the try block (line 465)
        with patch("json.loads", side_effect=RuntimeError("json parse crash")):
            with pytest.raises(ValueError, match="Failed to load model from"):
                load_model(model_path)


def test_register_keras_serializable_empty_package() -> None:
    """Test register_keras_serializable with empty package string."""

    @register_keras_serializable(package="", name="MyNoPackageCls")
    class NoPackageCls:
        """Dummy class without package."""

        pass

    assert get_registered_name(NoPackageCls) == "MyNoPackageCls"


def test_serialize_and_deserialize_nested_and_lists() -> None:
    """Test nested configs, lists, tuples, custom classes in serialization/deserialization."""

    class NestedObject:
        """Class with get_config method."""

        def get_config(self) -> dict[str, Any]:
            """Return config dictionary."""
            return {"inner_val": 42}

    # serialize_keras_object with dict and list/tuple
    obj_dict = {"layer": NestedObject(), "list_items": [NestedObject(), 10]}
    ser_dict = serialize_keras_object(obj_dict)
    assert isinstance(ser_dict, dict)
    assert ser_dict["layer"] == {"inner_val": 42}
    assert ser_dict["list_items"][0] == {"inner_val": 42}

    ser_list = serialize_keras_object([NestedObject(), (NestedObject(),)])
    assert isinstance(ser_list, list)
    assert ser_list[0] == {"inner_val": 42}

    # Test _serialize_nested_config fallback on an object without get_config, not dict/list/primitive (line 620)
    class CustomArbitrary:
        """Arbitrary object."""

        pass

    arbitrary_inst = CustomArbitrary()
    assert _serialize_nested_config(arbitrary_inst) is arbitrary_inst

    # Deserialization with custom_objects dict and non-callable class
    dummy_constant = "CONSTANT_OBJ"
    config = {
        "class_name": "MyConstant",
        "config": {},
    }
    res = deserialize_keras_object(config, custom_objects={"MyConstant": dummy_constant})
    assert res == dummy_constant

    # Deserialization of list / tuple
    list_config = [{"class_name": "MyConstant", "config": {}}, 123]
    res_list = deserialize_keras_object(list_config, custom_objects={"MyConstant": dummy_constant})
    assert res_list == [dummy_constant, 123]


def test_saved_model_topology_branches() -> None:
    """Test SavedModel with model.graph, model.to_json, model.get_config, and empty pb bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. model with .graph
        class ModelWithGraph:
            def __init__(self) -> None:
                self.graph = IRGraph(name="test_graph")

        sm1 = SavedModel(model=ModelWithGraph())
        save_path1 = os.path.join(tmpdir, "model1")
        sm1.save(save_path1)
        loaded1 = SavedModel.load(save_path1)
        assert loaded1.model is not None

        # 2. model with .to_json()
        class ModelWithToJson:
            def to_json(self) -> str:
                return json.dumps({"class_name": "DenseModel"})

        sm2 = SavedModel(model=ModelWithToJson())
        save_path2 = os.path.join(tmpdir, "model2")
        sm2.save(save_path2)
        loaded2 = SavedModel.load(save_path2)
        assert loaded2.model is not None

        # 3. model with .get_config() (line 835)
        class ModelWithGetConfig:
            def get_config(self) -> dict[str, Any]:
                return {"class_name": "ConfigOnlyModel"}

        sm3 = SavedModel(model=ModelWithGetConfig())
        save_path3 = os.path.join(tmpdir, "model3")
        sm3.save(save_path3)
        loaded3 = SavedModel.load(save_path3)
        assert loaded3.model is not None

        # 4. model that is None or has none of graph/to_json/get_config (line 835 false branch)
        class EmptyModel:
            pass

        sm4 = SavedModel(model=EmptyModel())
        save_path4 = os.path.join(tmpdir, "model4")
        sm4.save(save_path4)
        loaded4 = SavedModel.load(save_path4)
        assert loaded4.model is not None

        # 5. SavedModel.load missing directory or file
        with pytest.raises(FileNotFoundError, match="SavedModel directory"):
            SavedModel.load(os.path.join(tmpdir, "nonexistent_dir"))

        empty_dir = os.path.join(tmpdir, "empty_dir")
        os.makedirs(empty_dir)
        with pytest.raises(FileNotFoundError, match="SavedModel proto"):
            SavedModel.load(empty_dir)

        # 6. SavedModel.load with 0 byte saved_model.pb
        empty_pb_dir = os.path.join(tmpdir, "empty_pb_dir")
        os.makedirs(empty_pb_dir)
        with open(os.path.join(empty_pb_dir, "saved_model.pb"), "wb") as f:
            f.write(b"")
        loaded_empty = SavedModel.load(empty_pb_dir)
        assert loaded_empty.model is not None


def test_read_fingerprint_dir_and_file() -> None:
    """Test read_fingerprint on directories without fingerprint.pb and with files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Directory with files but without fingerprint.pb
        sub_dir = os.path.join(tmpdir, "checkpoint_dir")
        os.makedirs(sub_dir)
        with open(os.path.join(sub_dir, "weights.bin"), "wb") as f:
            f.write(b"weight_bytes_12345")

        fp1 = read_fingerprint(sub_dir)
        assert isinstance(fp1, str)
        assert len(fp1) == 64

        # Directory with empty fingerprint.pb -> falls back to hashing files
        with open(os.path.join(sub_dir, "fingerprint.pb"), "w", encoding="utf-8") as f:
            f.write("   \n")

        fp2 = read_fingerprint(sub_dir)
        assert fp2 == fp1


def test_load_checkpoint_dir_formats_and_validations() -> None:
    """Test run_restore_ops with safetensors, h5, target_model shape validation, and variables dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "ckpt_dir")
        os.makedirs(ckpt_dir)

        # Save numpy weights in checkpoint directory
        np.save(os.path.join(ckpt_dir, "w1.npy"), np.ones((2, 3), dtype=np.float32))
        np.save(os.path.join(ckpt_dir, "w2.npy"), np.zeros((4, 4), dtype=np.float32))

        from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
        from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat

        SafetensorsWeightFormat().save({"st_w": np.ones((1,), dtype=np.float32)}, os.path.join(ckpt_dir, "model.safetensors"))
        H5WeightFormat().save({"h5_w": np.ones((1,), dtype=np.float32)}, os.path.join(ckpt_dir, "model.h5"))

        restored = run_restore_ops(ckpt_dir)
        assert "w1" in restored
        assert "w2" in restored
        assert "st_w" in restored
        assert "h5_w" in restored

        # 2. Validate against target_model with assign method and without assign method
        class DummyWeightWithAssign:
            def __init__(self, name: str, shape: tuple[int, ...]) -> None:
                self.name = name
                self.shape = shape
                self.assigned_val: np.ndarray | None = None

            def assign(self, val: np.ndarray) -> None:
                self.assigned_val = val

        class DummyWeightNoAssign:
            def __init__(self, name: str, shape: tuple[int, ...]) -> None:
                self.name = name
                self.shape = shape

        class DummyWeightNoNameOrMissing:
            def __init__(self) -> None:
                self.name = "missing_var"
                self.shape = (1, 1)

        class DummyModel:
            def __init__(self) -> None:
                self.weights = [
                    DummyWeightWithAssign("w1", (2, 3)),
                    DummyWeightNoAssign("w2", (4, 4)),
                    DummyWeightNoNameOrMissing(),
                ]

        model_ok = DummyModel()
        run_restore_ops(ckpt_dir, target_model=model_ok)
        assert model_ok.weights[0].assigned_val is not None

        # 3. Validate shape mismatch in target_model
        class DummyModelMismatch:
            def __init__(self) -> None:
                self.weights = [DummyWeightWithAssign("w1", (5, 5))]

        with pytest.raises(ValueError, match="Shape mismatch restoring variable 'w1'"):
            run_restore_ops(ckpt_dir, target_model=DummyModelMismatch())

        # 4. Validate variables dict matching shape, mismatching shape, and missing key
        class DummyVar:
            def __init__(self, shape: tuple[int, ...]) -> None:
                self.shape = shape

        # Both present variable and missing variable
        run_restore_ops(ckpt_dir, variables={"w1": DummyVar((2, 3)), "missing_key": DummyVar((10,))})

        # Mismatched shape in variables
        with pytest.raises(ValueError, match="Shape mismatch restoring variable 'w1'"):
            run_restore_ops(ckpt_dir, variables={"w1": DummyVar((9, 9))})
