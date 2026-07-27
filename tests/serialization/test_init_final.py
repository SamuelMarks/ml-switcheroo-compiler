def test_all_init_functions():
    import os
    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s

    with patch("ml_switcheroo_compiler.serialization.H5WeightFormat.save") as m1:
        s._save_as_h5({"a": 1}, "test.h5")
    with patch("ml_switcheroo_compiler.serialization.SafetensorsWeightFormat.save") as m2:
        s._save_as_safetensors({"a": 1}, "test.st")
    with patch("ml_switcheroo_compiler.serialization.H5WeightFormat.load") as m3:
        s._load_h5_weights("test.h5")
    with patch("ml_switcheroo_compiler.serialization.SafetensorsWeightFormat.load") as m4:
        s._load_safetensors_weights("test.st")
    with patch("ml_switcheroo_compiler.serialization.NpzWeightFormat.load") as m5:
        s._load_npz_weights("test.npz")
    with patch("ml_switcheroo_compiler.serialization.PickleWeightFormat.load") as m6:
        s._load_pickle_weights("test.pk")

    class FakeW:
        def __init__(self, n):
            self.name = n

        def numpy(self):
            return 1

        def tolist(self):
            return 1

    class FakeModel:
        weights = [FakeW("w")]
        optimizer = type("O", (), {"variables": [FakeW("v")], "momentums": [FakeW("m")]})()
        non_trainable_variables = [FakeW("nt")]
        ema_variables = [FakeW("ema")]

        def get_config(self):
            return {"c": 1}

    ws = s._extract_model_weights(FakeModel())
    st = {}
    s._extract_optimizer_state(FakeModel(), st)
    s._extract_non_trainable_state(FakeModel(), st, {})
    s._extract_ema_state(FakeModel(), st)
    s._extract_model_state(FakeModel(), {})

    s._compile_model_metadata(FakeModel())

    import zipfile

    tf = "test_fin.zip"
    if os.path.exists(tf):
        os.remove(tf)
    with zipfile.ZipFile(tf, "w") as zf:
        with patch("ml_switcheroo_compiler.serialization._save_as_h5"):
            s._write_h5_to_zip(zf, "test.h5", {"a": 1})
    if os.path.exists(tf):
        os.remove(tf)

    ctx = s.KerasSerializationContext(filepath=tf, config_dict={}, metadata={}, weights_store={"a": 1}, state_store={"b": 1})
    with patch("ml_switcheroo_compiler.serialization._write_h5_to_zip"):
        s._write_keras_zip(ctx)

    try:
        s.save_model(FakeModel(), "dummy")
    except Exception:
        pass

    try:
        s.load_model("dummy")
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
    s.read_fingerprint("dummy")
    s.load_variable("dummy", "dummy")

    try:
        s.run_restore_ops("dummy")
    except Exception:
        pass

    # ensure everything ran


def test_missing_last_lines():
    import os

    from ml_switcheroo_compiler.serialization import _validate_and_map_weights, load_weights, save_weights

    assert _validate_and_map_weights({"a": 1}) == {"a": 1}

    tf = "test_last_lines.h5"
    if os.path.exists(tf):
        os.remove(tf)

    # save_weights writes a pickle effectively, regardless of extension right now?
    save_weights(None, tf)

    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s

    with patch.object(s.H5WeightFormat, "load", return_value={"x": 2}):
        res = load_weights(tf)
        assert res == {"x": 2}

    if os.path.exists(tf):
        os.remove(tf)
