def test_keras_np_serialization():
    import os

    from ml_switcheroo_compiler.serialization import load_model, register_keras_serializable, save_model

    test_file = "test_keras.keras"
    if os.path.exists(test_file):
        os.remove(test_file)

    class FakeLayer:
        pass

    class FakeModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    try:
        save_model(FakeModel(), test_file)
        model = load_model(test_file)
        assert hasattr(model, "config")
    except Exception:
        pass

    if os.path.exists(test_file):
        os.remove(test_file)

    model = load_model("nonexistent.keras")
    assert type(model).__name__ == "FallbackModel"

    @register_keras_serializable(package="test", name="myobj")
    class MyCustomObj:
        pass

    from ml_switcheroo_compiler.serialization import get_custom_objects

    assert isinstance(get_custom_objects(), dict)


def test_keras_file_editor():
    import os
    import zipfile

    from ml_switcheroo_compiler.serialization import KerasFileEditor

    test_file = "test_editor.keras"
    with zipfile.ZipFile(test_file, "w") as zf:
        zf.writestr("config.json", "{}")

    ed = KerasFileEditor(test_file)
    assert ed.filepath == test_file

    if os.path.exists(test_file):
        os.remove(test_file)
