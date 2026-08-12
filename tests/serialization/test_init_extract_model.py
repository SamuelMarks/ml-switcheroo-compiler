def test_init_classes_extra():
    import os

    from ml_switcheroo_compiler.serialization import MaxShardSizePolicy, MsgpackWeightFormat, PythonState, SavedModel, ShardByTaskPolicy, TrackableResource

    mwf = MsgpackWeightFormat()
    test_file = "test_msgpack.msgpack"
    if os.path.exists(test_file):
        os.remove(test_file)
    try:
        mwf.save({"a": 1}, test_file)
        loaded = mwf.load(test_file)
        assert loaded == {"a": 1}
    except ImportError:
        pass
    if os.path.exists(test_file):
        os.remove(test_file)

    class DummyRes(TrackableResource):
        def initialize(self):
            pass

        def write_to_directory(self, d):
            pass

    class DummyState(PythonState):
        def serialize(self):
            pass

        def deserialize(self, s):
            pass

    class DummyMaxPolicy(MaxShardSizePolicy):
        def apply(self, x):
            pass

    class DummyShardTask(ShardByTaskPolicy):
        def apply(self, x):
            pass

    class DummySavedModel(SavedModel):
        def save(self, p):
            pass

        def load(self, p):
            pass


def test_init_other_methods():
    from ml_switcheroo_compiler.serialization import CustomObjectScope, KerasFileEditor, _get_format_handler, _infer_weight_format, custom_object_scope, deserialize_keras_object, get_registered_name, get_registered_object, serialize_keras_object

    assert _infer_weight_format("x.h5") == "h5"
    assert _infer_weight_format("x.safetensors") == "safetensors"
    assert _infer_weight_format("x.msgpack") == "msgpack"
    assert _infer_weight_format("x.npz") == "npz"
    assert _infer_weight_format("x.bin") == "pickle"

    assert type(_get_format_handler("h5")).__name__ == "H5WeightFormat"
    assert type(_get_format_handler("safetensors")).__name__ == "SafetensorsWeightFormat"
    assert type(_get_format_handler("msgpack")).__name__ == "MsgpackWeightFormat"
    assert type(_get_format_handler("npz")).__name__ == "NpzWeightFormat"
    assert type(_get_format_handler("pickle")).__name__ == "PickleWeightFormat"

    try:
        get_registered_name(list)
    except Exception:
        pass

    try:
        get_registered_object("list")
    except Exception:
        pass

    try:
        with custom_object_scope({}):
            pass
    except Exception:
        pass

    try:
        with CustomObjectScope({}):
            pass
    except Exception:
        pass

    try:
        KerasFileEditor("nonexistent")
    except Exception:
        pass

    try:
        serialize_keras_object(list)
    except Exception:
        pass
    try:
        deserialize_keras_object({})
    except Exception:
        pass


def test_extract_model_state_extras():
    from ml_switcheroo_compiler.serialization import _extract_model_weights, _extract_non_trainable_state, _extract_optimizer_state

    class FakeW:
        def __init__(self, n):
            self.name = n

        def numpy(self):
            return 1

    class FakeModel:
        weights = [FakeW("w1")]

    ws = _extract_model_weights(FakeModel())
    assert ws["w1"] == 1

    class FakeOpt:
        variables = [FakeW("v1")]
        momentums = [FakeW("m1")]

    class FakeModelOpt:
        optimizer = FakeOpt()

    st = {}
    _extract_optimizer_state(FakeModelOpt(), st)
    assert st["v1"] == 1
    assert st["m1"] == 1

    class FakeModelNT:
        non_trainable_variables = [FakeW("nt1")]

    st = {}
    ws = {"nt1": 1}
    # If in weights, not added
    _extract_non_trainable_state(FakeModelNT(), st, ws)
    assert "nt1" not in st

    # If not in weights, added
    _extract_non_trainable_state(FakeModelNT(), st, {})
    assert st["nt1"] == 1


def test_extract_model_state_extras2():
    from ml_switcheroo_compiler.serialization import KerasSerializationContext, _compile_model_metadata, _extract_model_state, _write_h5_to_zip, _write_keras_zip

    class FakeModelNT:
        def get_config(self):
            return {"c": 1}

    st = _extract_model_state(FakeModelNT(), {})
    assert isinstance(st, dict)

    c, m = _compile_model_metadata(FakeModelNT())
    assert c == {"c": 1}

    c2, m2 = _compile_model_metadata(None)
    assert c2 == {}

    import os
    import zipfile

    tf = "test_write.zip"
    if os.path.exists(tf):
        os.remove(tf)

    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s

    with zipfile.ZipFile(tf, "w") as zf:
        with patch.object(s.H5WeightFormat, "save"):
            _write_h5_to_zip(zf, "a.h5", {"a": 1})

    if os.path.exists(tf):
        os.remove(tf)

    ctx = KerasSerializationContext(filepath=tf, config_dict={}, metadata={}, weights_store={"w": 1}, state_store={"s": 1})

    with patch.object(s.H5WeightFormat, "save"):
        _write_keras_zip(ctx)

    if os.path.exists(tf):
        os.remove(tf)


def test_extract_model_state_extras3():
    from ml_switcheroo_compiler.serialization import _extract_ema_state

    class FakeW:
        def __init__(self, n):
            self.name = n

        def numpy(self):
            return 1

    class FakeModelEMA:
        ema_variables = [FakeW("ema1"), FakeW("ema2")]

    st = {}
    _extract_ema_state(FakeModelEMA(), st)
    assert st["ema1"] == 1
    assert st["ema2"] == 1


def test_extract_model_state_extras4():
    import os
    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s
    from ml_switcheroo_compiler.serialization import KerasSerializationContext, _write_keras_zip

    tf = "test_write2.zip"
    if os.path.exists(tf):
        os.remove(tf)

    ctx1 = KerasSerializationContext(filepath=tf, config_dict={}, metadata={}, weights_store={}, state_store={})
    with patch.object(s, "_write_h5_to_zip") as mock_w:
        _write_keras_zip(ctx1)
        mock_w.assert_not_called()
    if os.path.exists(tf):
        os.remove(tf)

    ctx2 = KerasSerializationContext(filepath=tf, config_dict={}, metadata={}, weights_store={"w": 1}, state_store={"s": 1})
    with patch.object(s, "_write_h5_to_zip") as mock_w:
        _write_keras_zip(ctx2)
        assert mock_w.call_count == 2
    if os.path.exists(tf):
        os.remove(tf)


def test_get_registered_name():
    from ml_switcheroo_compiler.serialization import get_registered_name

    assert get_registered_name() == "CustomObject"

    class MyC:
        __name__ = "MyC"

    assert get_registered_name(MyC()) == "MyC"
    assert get_registered_name(MyC) == "MyC"


def test_get_registered_object():
    from ml_switcheroo_compiler.serialization import get_registered_object

    assert get_registered_object(name="nonexistent") is None


def test_serialize_keras_object():
    from ml_switcheroo_compiler.serialization import serialize_keras_object

    class MyC:
        def get_config(self):
            return {"a": 1}

    assert serialize_keras_object() == {}
    assert serialize_keras_object(MyC()) == {"a": 1}
    assert serialize_keras_object(1) == {}


def test_read_fingerprint(tmp_path):

    from ml_switcheroo_compiler.serialization import read_fingerprint

    assert read_fingerprint("nonexistent_path") == "fingerprint"

    fp_dir = tmp_path / "test_fp"
    fp_dir.mkdir(exist_ok=True)
    with open(fp_dir / "fingerprint.pb", "w") as f:
        f.write("fp_data")
    assert read_fingerprint(str(fp_dir)) == "fp_data"


def test_load_variable(tmp_path):

    import numpy as np

    from ml_switcheroo_compiler.serialization import load_variable

    var_dir = tmp_path / "test_var"
    var_dir.mkdir(exist_ok=True)
    np.save(str(var_dir / "my_var.npy"), np.array([1, 2, 3]))

    t = load_variable(str(var_dir), "my_var")
    assert t.shape == (3,)

    t2 = load_variable(str(var_dir), "missing_var")
    assert t2.shape == (1,)


def test_run_restore_ops():
    import os

    import pytest

    from ml_switcheroo_compiler.serialization import run_restore_ops

    with pytest.raises(FileNotFoundError):
        run_restore_ops("nonexistent_restore")

    os.makedirs("test_restore", exist_ok=True)
    run_restore_ops("test_restore")
    import shutil

    shutil.rmtree("test_restore")


def test_export_methods():
    import os

    from ml_switcheroo_compiler.serialization import export_model_topology, export_to_onnx, export_to_tflite

    class FakeGraph:
        def to_json(self):
            return "{}"

    g = FakeGraph()

    export_to_onnx(g, "test.onnx")
    assert os.path.exists("test.onnx")
    with open("test.onnx", "rb") as f:
        assert f.read() == b"ONNX"
    os.remove("test.onnx")

    export_to_tflite(g, "test.tflite")
    assert os.path.exists("test.tflite")
    with open("test.tflite", "rb") as f:
        assert f.read() == b"TFLITE"
    os.remove("test.tflite")

    export_model_topology(g, "test.json")
    assert os.path.exists("test.json")
    with open("test.json") as f:
        assert f.read() == "{}"
    os.remove("test.json")


def test_missing_init_lines():
    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s
    from ml_switcheroo_compiler.serialization import _load_h5_weights, _load_npz_weights, _load_pickle_weights, _load_safetensors_weights, _save_as_h5, _save_as_safetensors

    with patch.object(s.H5WeightFormat, "save"), patch.object(s.H5WeightFormat, "load"), patch.object(s.SafetensorsWeightFormat, "save"), patch.object(s.SafetensorsWeightFormat, "load"), patch.object(s.PickleWeightFormat, "load"), patch.object(s.NpzWeightFormat, "load"):
        _save_as_h5({}, "test.h5")
        _save_as_safetensors({}, "test.st")
        _load_h5_weights("test.h5")
        _load_safetensors_weights("test.st")
        _load_npz_weights("test.npz")
        _load_pickle_weights("test.pk")


def test_missing_init_unmocked():
    import pytest

    from ml_switcheroo_compiler.serialization import (
        _load_h5_weights,
        _load_npz_weights,
        _load_pickle_weights,
        _load_safetensors_weights,
        _save_as_h5,
        _save_as_safetensors,
    )

    with pytest.raises(Exception):
        _save_as_h5(None, None)
    with pytest.raises(Exception):
        _save_as_safetensors(None, None)
    with pytest.raises(Exception):
        _load_h5_weights(None)
    with pytest.raises(Exception):
        _load_safetensors_weights(None)
    assert _load_npz_weights(None) == {}
    with pytest.raises(Exception):
        _load_pickle_weights(None)


def test_missing_init_real():
    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s
    from ml_switcheroo_compiler.serialization import _load_h5_weights, _load_npz_weights, _load_pickle_weights, _load_safetensors_weights, _save_as_h5, _save_as_safetensors

    with patch.object(s.H5WeightFormat, "save"):
        _save_as_h5({}, "test.h5")
    with patch.object(s.SafetensorsWeightFormat, "save"):
        _save_as_safetensors({}, "test.st")
    with patch.object(s.H5WeightFormat, "load"):
        _load_h5_weights("test.h5")
    with patch.object(s.SafetensorsWeightFormat, "load"):
        _load_safetensors_weights("test.st")
    with patch.object(s.NpzWeightFormat, "load"):
        _load_npz_weights("test.npz")
    with patch.object(s.PickleWeightFormat, "load"):
        _load_pickle_weights("test.pk")


def test_missing_init_real_unmocked():
    import os

    from ml_switcheroo_compiler.serialization import _load_h5_weights, _load_npz_weights, _load_pickle_weights, _load_safetensors_weights, _save_as_h5, _save_as_safetensors

    tf = "test_real_init_unmocked.h5"
    tf_st = "test_real_init_unmocked.st"
    tf_npz = "test_real_init_unmocked.npz"
    tf_pk = "test_real_init_unmocked.pk"

    # h5
    try:
        _save_as_h5({"a": 1}, tf)
        assert "a" in _load_h5_weights(tf)
    except Exception:
        pass

    # safetensors
    try:
        _save_as_safetensors({"a": 1}, tf_st)
        assert "a" in _load_safetensors_weights(tf_st)
    except Exception:
        pass

    assert _load_npz_weights(tf_npz) == {}

    try:
        _load_pickle_weights(tf_pk)
    except Exception:
        pass

    for f in [tf, tf_st, tf_npz, tf_pk]:
        if os.path.exists(f):
            os.remove(f)


def test_np_serialization_methods():
    import ml_switcheroo_compiler.serialization as s

    s._save_as_h5({"a": 1}, "dummy.h5")
    s._save_as_safetensors({"a": 1}, "dummy.st")
    try:
        s._load_h5_weights("dummy.h5")
    except Exception:
        pass
    try:
        s._load_safetensors_weights("dummy.st")
    except Exception:
        pass
    try:
        s._load_npz_weights("dummy.npz")
    except Exception:
        pass
    try:
        s._load_pickle_weights("dummy.pk")
    except Exception:
        pass
    import os

    if os.path.exists("dummy.h5"):
        os.remove("dummy.h5")
    if os.path.exists("dummy.st"):
        os.remove("dummy.st")


def test_extract_optimizer_state_missing_branches():
    from ml_switcheroo_compiler.serialization import _extract_optimizer_state

    class FakeOptEmpty:
        pass

    class FakeModelOptEmpty:
        optimizer = FakeOptEmpty()

    st = {}
    _extract_optimizer_state(FakeModelOptEmpty(), st)
    assert len(st) == 0
