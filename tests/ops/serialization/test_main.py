# ruff: noqa: E501
import os
import tempfile
import zipfile

import pytest

from ml_switcheroo_compiler.serialization import (
    CustomObjectScope,
    KerasFileEditor,
    KerasSerializationContext,
    MaxShardSizePolicy,
    MsgpackWeightFormat,
    PythonState,
    SavedModel,
    ShardByTaskPolicy,
    TrackableResource,
    _compile_model_metadata,
    _extract_ema_state,
    _extract_model_state,
    _extract_model_weights,
    _extract_non_trainable_state,
    _extract_optimizer_state,
    _get_format_handler,
    _infer_weight_format,
    _load_h5_weights,
    _load_npz_weights,
    _load_pickle_weights,
    _load_safetensors_weights,
    _save_as_h5,
    _save_as_safetensors,
    _validate_and_map_weights,
    _write_h5_to_zip,
    _write_keras_zip,
    custom_object_scope,
    deserialize_keras_object,
    export_model_topology,
    export_to_onnx,
    export_to_tflite,
    get_custom_objects,
    get_registered_name,
    get_registered_object,
    graph_to_json,
    load_model,
    load_variable,
    load_weights,
    read_fingerprint,
    register_keras_serializable,
    run_restore_ops,
    save_model,
    save_weights,
    serialize_keras_object,
)


class MockModel:
    def __init__(self):

        class W:
            name = "w1"

            def numpy(self):
                return "w1_data"

        self.weights = [W()]

        class W2:
            def numpy(self):
                return "w2_data"

        self.weights.append(W2())

        class Opt:
            class V:
                name = "v1"

                def numpy(self):
                    return "v1_data"

            class M:
                name = "m1"

                def numpy(self):
                    return "m1_data"

            variables = [V()]
            momentums = [M()]

        self.optimizer = Opt()

        class V2:
            def numpy(self):
                return "v2_data"

        self.optimizer.variables.append(V2())

        class M2:
            def numpy(self):
                return "m2_data"

        self.optimizer.momentums.append(M2())

        class NT:
            name = "nt1"

            def numpy(self):
                return "nt1_data"

        self.non_trainable_variables = [NT()]

        class NT2:
            def numpy(self):
                return "nt2_data"

        self.non_trainable_variables.append(NT2())

        class EMA:
            name = "ema1"

            def numpy(self):
                return "ema1_data"

        self.ema_variables = [EMA()]

        class EMA2:
            def numpy(self):
                return "ema2_data"

        self.ema_variables.append(EMA2())

    def get_config(self):
        return {"conf": "mock"}


def test_msgpack_weight_format(mocker):
    fmt = MsgpackWeightFormat()
    mocker.patch.dict("sys.modules", {"msgpack": None})
    with pytest.raises(ImportError):
        fmt.load("test.msgpack")
    with pytest.raises(ImportError):
        fmt.save({}, "test.msgpack")
    mock_msgpack = mocker.MagicMock()
    mock_msgpack.unpackb.return_value = {"a": 1}
    mock_msgpack.packb.return_value = b"pack"
    mocker.patch.dict("sys.modules", {"msgpack": mock_msgpack})
    with tempfile.NamedTemporaryFile() as f:
        fmt.save({"a": 1}, f.name)
        assert fmt.load(f.name) == {"a": 1}


def test_infer_weight_format():
    assert _infer_weight_format("a.h5") == "h5"
    assert _infer_weight_format("a.safetensors") == "safetensors"
    assert _infer_weight_format("a.msgpack") == "msgpack"
    assert _infer_weight_format("a.npz") == "npz"
    assert _infer_weight_format("a.pkl") == "pickle"


def test_get_format_handler():
    assert type(_get_format_handler("h5")).__name__ == "H5WeightFormat"
    assert type(_get_format_handler("safetensors")).__name__ == "SafetensorsWeightFormat"
    assert type(_get_format_handler("msgpack")).__name__ == "MsgpackWeightFormat"
    assert type(_get_format_handler("npz")).__name__ == "NpzWeightFormat"
    assert type(_get_format_handler("pickle")).__name__ == "PickleWeightFormat"


def test_save_load_helpers(mocker):
    mocker.patch("ml_switcheroo_compiler.serialization.H5WeightFormat.save")
    _save_as_h5({}, "test.h5")
    mocker.patch("ml_switcheroo_compiler.serialization.SafetensorsWeightFormat.save")
    _save_as_safetensors({}, "test.st")
    mocker.patch("ml_switcheroo_compiler.serialization.H5WeightFormat.load", return_value={})
    assert _load_h5_weights("test.h5") == {}
    mocker.patch("ml_switcheroo_compiler.serialization.SafetensorsWeightFormat.load", return_value={})
    assert _load_safetensors_weights("test.st") == {}
    mocker.patch("ml_switcheroo_compiler.serialization.NpzWeightFormat.load", return_value={})
    assert _load_npz_weights("test.npz") == {}
    mocker.patch("ml_switcheroo_compiler.serialization.PickleWeightFormat.load", return_value={})
    assert _load_pickle_weights("test.pkl") == {}


def test_validate_and_map_weights():
    assert _validate_and_map_weights({"a": 1}) == {"a": 1}


def test_load_save_weights(mocker):
    mocker.patch("ml_switcheroo_compiler.serialization.PickleWeightFormat.load", return_value={"a": 1})
    assert load_weights("test.pkl") == {"a": 1}
    with tempfile.NamedTemporaryFile() as f:
        save_weights(MockModel(), f.name)
        assert os.path.exists(f.name)


def test_export_funcs(mocker):
    mock_graph = mocker.MagicMock()
    mock_graph.to_json.return_value = "{}"
    with tempfile.NamedTemporaryFile() as f:
        export_to_onnx(mock_graph, f.name)
        export_to_tflite(mock_graph, f.name)
        export_model_topology(mock_graph, f.name)
    assert graph_to_json(mock_graph) == "{}"


def test_extract_model_state(mocker):
    mocker.patch("ml_switcheroo_compiler.serialization.to_numpy", side_effect=lambda x: x.numpy())

    class MockModelNoConfig:
        pass

    m2 = MockModelNoConfig()
    _compile_model_metadata(m2)
    m = MockModel()
    weights = _extract_model_weights(m)
    assert weights == {"w1": "w1_data", "weight_1": "w2_data"}
    state = {}
    _extract_optimizer_state(m, state)
    assert state == {"v1": "v1_data", "opt_state_1": "v2_data", "m1": "m1_data", "momentum_1": "m2_data"}
    state = {}
    _extract_non_trainable_state(m, state, weights)

    class NT2:
        name = "nt2"

        def numpy(self):
            return "nt2_data"

    _extract_non_trainable_state(m, state, {"nt2": "nt2_data"})
    pass
    state = {}
    _extract_ema_state(m, state)
    pass
    state = _extract_model_state(m, weights)

    class MockModelNoOptimizer:
        pass

    _extract_optimizer_state(MockModelNoOptimizer(), {})
    pass


def test_compile_model_metadata():

    class MockModelNoConfig:
        pass

    m2 = MockModelNoConfig()
    _compile_model_metadata(m2)
    m = MockModel()
    (config, meta) = _compile_model_metadata(m)
    assert config == {"conf": "mock"}
    assert "keras_version" in meta


def test_write_h5_to_zip(mocker, tmp_path):
    mocker.patch("ml_switcheroo_compiler.serialization._save_as_h5")
    with tempfile.NamedTemporaryFile() as f:
        with zipfile.ZipFile(f.name, "w") as zf:
            _write_h5_to_zip(zf, "test.h5", {})

    keras_path = str(tmp_path / "model_bundle.keras")
    ctx = KerasSerializationContext(keras_path, {}, {}, {}, {})
    _write_keras_zip(ctx)


def test_keras_zip_save_load(mocker):
    mocker.patch("ml_switcheroo_compiler.serialization._save_as_h5")
    mocker.patch("ml_switcheroo_compiler.serialization.to_numpy", side_effect=lambda x: x.numpy())

    class MockModelNoConfig:
        pass

    m2 = MockModelNoConfig()
    _compile_model_metadata(m2)
    m = MockModel()
    with tempfile.NamedTemporaryFile() as f:
        save_model(m, f.name)
        m_loaded = load_model(f.name)
        assert m_loaded.config == {"conf": "mock"}


def test_load_model_exception():
    assert load_model("nonexistent.keras") is not None


def test_register_keras_serializable():

    @register_keras_serializable("Pkg", "Name")
    class TestObj:
        pass

    assert TestObj.__name__ == "TestObj"


def test_custom_object_scope():
    with custom_object_scope({"a": 1}) as scope:
        assert scope.custom_objects == {"a": 1}
    with CustomObjectScope() as scope:
        pass


def test_keras_file_editor():
    editor = KerasFileEditor("test.keras")
    assert editor.filepath == "test.keras"


def test_deserialize_serialize_keras_object():
    assert deserialize_keras_object({"a": 1}) == {"a": 1}
    assert deserialize_keras_object(a=1) == {"a": 1}

    class Obj:
        def get_config(self):
            return {"a": 1}

    assert serialize_keras_object(Obj()) == {"a": 1}
    assert serialize_keras_object() == {}


def test_custom_objects_registry():
    assert isinstance(get_custom_objects(), dict)

    class TestObj:
        pass

    assert get_registered_name(TestObj) == "TestObj"
    assert get_registered_name() == "CustomObject"

    class A:
        __name__ = "A_name"

    assert get_registered_name(A) == "A"
    assert get_registered_object(name="test") is None


def test_classes_init():
    assert TrackableResource() is not None
    assert PythonState().state == {}
    assert MaxShardSizePolicy(10).max_shard_size == 10
    assert ShardByTaskPolicy().policy == "task"


def test_saved_model():
    sm = SavedModel()
    with tempfile.TemporaryDirectory() as td:
        sm.save(td)
        assert os.path.exists(os.path.join(td, "saved_model.pb"))
    assert SavedModel.load("test") is not None


def test_read_fingerprint():
    with tempfile.TemporaryDirectory() as td:
        assert read_fingerprint(td) == "fingerprint"
        with open(os.path.join(td, "fingerprint.pb"), "w") as f:
            f.write("fp_data")
        assert read_fingerprint(td) == "fp_data"


def test_load_variable():
    with tempfile.TemporaryDirectory() as td:
        res = load_variable(td, "var1")
        assert res.config.shape == (1,)
        import numpy as np

        np.save(os.path.join(td, "var1.npy"), np.array([1, 2]))
        res2 = load_variable(td, "var1")
        assert res2.config.shape == (2,)


def test_run_restore_ops():
    with pytest.raises(FileNotFoundError):
        run_restore_ops("nonexistent_path")
    with tempfile.TemporaryDirectory() as td:
        run_restore_ops(td)
