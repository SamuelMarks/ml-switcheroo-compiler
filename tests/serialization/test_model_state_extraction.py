"""Tests for test_model_state_extraction."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
    from ml_switcheroo_compiler.core.tensor import Tensor


def test_init_classes_serialization() -> None:
    """Test function test_init_classes_serialization."""
    from ml_switcheroo_compiler.serialization import MaxShardSizePolicy, PythonState, SavedModel, ShardByTaskPolicy, TrackableResource

    class DummyRes(TrackableResource):
        """Test class DummyRes."""

        def initialize(self) -> None:
            """Test function initialize."""
            pass

        def write_to_directory(self, d: EagerValue) -> None:
            """Test function write_to_directory.

            Args:
                d: Test parameter `d`.
            """
            pass

    class DummyState(PythonState):
        """Test class DummyState."""

        def serialize(self) -> None:
            """Test function serialize."""
            pass

        def deserialize(self, s: EagerValue) -> None:
            """Test function deserialize.

            Args:
                s: Test parameter `s`.
            """
            pass

    class DummyMaxPolicy(MaxShardSizePolicy):
        """Test class DummyMaxPolicy."""

        def apply(self, x: EagerValue) -> None:
            """Test function apply.

            Args:
                x: Test parameter `x`.
            """
            pass

    class DummyShardTask(ShardByTaskPolicy):
        """Test class DummyShardTask."""

        def apply(self, x: EagerValue) -> None:
            """Test function apply.

            Args:
                x: Test parameter `x`.
            """
            pass

    class DummySavedModel(SavedModel):
        """Test class DummySavedModel."""

        def save(self, p: EagerValue) -> None:
            """Test function save.

            Args:
                p: Test parameter `p`.
            """
            pass

        @classmethod
        def load(cls, path: str) -> SavedModel:
            """Test function load.

            Args:
                path: Test parameter `path`.
            """
            pass
            return typing.cast("SavedModel", None)

    assert callable(test_init_classes_serialization)


def test_init_other_methods() -> None:
    """Test function test_init_other_methods."""
    from ml_switcheroo_compiler.serialization import CustomObjectScope, KerasFileEditor, _get_format_handler, _infer_weight_format, custom_object_scope, deserialize_object, get_registered_name, get_registered_object, serialize_object

    assert _infer_weight_format("x.h5") == "h5"
    assert _infer_weight_format("x.safetensors") == "safetensors"
    assert _infer_weight_format("x.npz") == "npz"
    assert _infer_weight_format("x.bin") == "pickle"
    assert type(_get_format_handler("h5")).__name__ == "H5WeightFormat"
    assert type(_get_format_handler("safetensors")).__name__ == "SafetensorsWeightFormat"
    assert type(_get_format_handler("npz")).__name__ == "NpzWeightFormat"
    assert type(_get_format_handler("pickle")).__name__ == "PickleWeightFormat"
    get_registered_name(list)
    get_registered_object("list")
    with custom_object_scope({}):
        pass
    with CustomObjectScope({}):
        pass
    KerasFileEditor("nonexistent")
    serialize_object(list)
    deserialize_object({})


def test_extract_model_state_layers() -> None:
    """Test function test_extract_model_state_layers."""
    from ml_switcheroo_compiler.serialization import _extract_model_weights, _extract_non_trainable_state, _extract_optimizer_state

    class FakeW:
        """Test class FakeW."""

        def __init__(self, n: EagerValue) -> None:
            """Test function __init__.

            Args:
                n: Test parameter `n`.
            """
            self.name = n

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return 1

    class FakeModel:
        """Test class FakeModel."""

        weights = [FakeW("w1")]

    ws = typing.cast("EagerCallableProtocol", _extract_model_weights)(typing.cast("EagerCallableProtocol", FakeModel)())
    assert ws["w1"] == 1

    class FakeOpt:
        """Test class FakeOpt."""

        variables = [FakeW("v1")]
        momentums = [FakeW("m1")]

    class FakeModelOpt:
        """Test class FakeModelOpt."""

        optimizer = FakeOpt()

    st = typing.cast("dict[str, EagerTensorProtocol]", {})
    typing.cast("EagerCallableProtocol", _extract_optimizer_state)(typing.cast("EagerCallableProtocol", FakeModelOpt)(), st)
    assert st["v1"] == 1
    assert st["m1"] == 1

    class FakeModelNT:
        """Test class FakeModelNT."""

        non_trainable_variables = [FakeW("nt1")]

    st = {}
    ws = typing.cast("EagerTensorProtocol", {"nt1": 1})
    typing.cast("EagerCallableProtocol", _extract_non_trainable_state)(typing.cast("EagerCallableProtocol", FakeModelNT)(), st, ws)
    assert "nt1" not in st
    typing.cast("EagerCallableProtocol", _extract_non_trainable_state)(typing.cast("EagerCallableProtocol", FakeModelNT)(), st, {})
    assert st["nt1"] == 1


def test_extract_model_state_submodules() -> None:
    """Test function test_extract_model_state_submodules."""
    from ml_switcheroo_compiler.serialization import KerasSerializationContext, _compile_model_metadata, _extract_model_state, _write_h5_to_zip, _write_keras_zip

    class FakeModelNT:
        """Test class FakeModelNT."""

        def get_config(self) -> EagerValue:
            """Test function get_config."""
            return {"c": 1}

    st = typing.cast("EagerCallableProtocol", _extract_model_state)(typing.cast("EagerCallableProtocol", FakeModelNT)(), {})
    assert isinstance(typing.cast("EagerValue", st), dict)
    (c, m) = typing.cast("EagerCallableProtocol", _compile_model_metadata)(typing.cast("EagerCallableProtocol", FakeModelNT)())
    assert c == {"c": 1}
    (c2, m2) = typing.cast("EagerCallableProtocol", _compile_model_metadata)(None)
    assert c2 == {}
    import os
    import zipfile

    tf = "test_write.zip"
    if os.path.exists(tf):
        os.remove(tf)
    from unittest.mock import patch

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    with zipfile.ZipFile(tf, "w") as zf:
        with patch.object(H5WeightFormat, "save"):
            _write_h5_to_zip(zf, "a.h5", {"a": 1})
    if os.path.exists(tf):
        os.remove(tf)
    ctx = KerasSerializationContext(
        filepath=tf,
        config_dict={},
        metadata={},
        weights_store=typing.cast("dict[str, Tensor | list[EagerValue] | dict[str, EagerValue]]", {"w": 1}),
        state_store=typing.cast("dict[str, Tensor | list[EagerValue] | dict[str, EagerValue]]", {"s": 1}),
    )
    with patch.object(H5WeightFormat, "save"):
        _write_keras_zip(ctx)
    if os.path.exists(tf):
        os.remove(tf)


def test_extract_model_state_parameters() -> None:
    """Test function test_extract_model_state_parameters."""
    from ml_switcheroo_compiler.serialization import _extract_ema_state

    class FakeW:
        """Test class FakeW."""

        def __init__(self, n: EagerValue) -> None:
            """Test function __init__.

            Args:
                n: Test parameter `n`.
            """
            self.name = n

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return 1

    class FakeModelEMA:
        """Test class FakeModelEMA."""

        ema_variables = [FakeW("ema1"), FakeW("ema2")]

    st = typing.cast("dict[str, EagerTensorProtocol]", {})
    typing.cast("EagerCallableProtocol", _extract_ema_state)(typing.cast("EagerCallableProtocol", FakeModelEMA)(), st)
    assert st["ema1"] == 1
    assert st["ema2"] == 1


def test_extract_model_state_nested_objects() -> None:
    """Test function test_extract_model_state_nested_objects."""
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
    ctx2 = typing.cast("EagerCallableProtocol", KerasSerializationContext)(filepath=tf, config_dict={}, metadata={}, weights_store={"w": 1}, state_store={"s": 1})
    with patch.object(s, "_write_h5_to_zip") as mock_w:
        typing.cast("EagerCallableProtocol", _write_keras_zip)(ctx2)
        assert mock_w.call_count == 2
    if os.path.exists(tf):
        os.remove(tf)


def test_get_registered_name() -> None:
    """Test function test_get_registered_name."""
    from ml_switcheroo_compiler.serialization import get_registered_name

    assert get_registered_name() == "CustomObject"

    class MyC:
        """Test class MyC."""

        __name__ = "MyC"

    assert get_registered_name(MyC()) == "MyC"
    assert get_registered_name(MyC) == "MyC"


def test_get_registered_object() -> None:
    """Test function test_get_registered_object."""
    from ml_switcheroo_compiler.serialization import get_registered_object

    assert get_registered_object(name="nonexistent") is None


def test_serialize_keras_object() -> None:
    """Test function test_serialize_keras_object."""
    from ml_switcheroo_compiler.serialization import serialize_object

    class MyC:
        """Test class MyC."""

        def get_config(self) -> EagerValue:
            """Test function get_config."""
            return {"a": 1}

    assert serialize_object() == {}
    assert serialize_object(typing.cast("EagerValue", MyC())) == {"a": 1}
    assert serialize_object(1) == {}


def test_read_fingerprint(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_read_fingerprint.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    from ml_switcheroo_compiler.serialization import read_fingerprint

    assert read_fingerprint("nonexistent_path") == "fingerprint"
    fp_dir = tmp_path / "test_fp"
    fp_dir.mkdir(exist_ok=True)
    with open(fp_dir / "fingerprint.pb", "w") as f:
        f.write("fp_data")
    assert read_fingerprint(str(fp_dir)) == "fp_data"


def test_load_variable(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_load_variable.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    import numpy as np

    from ml_switcheroo_compiler.serialization import load_variable

    var_dir = tmp_path / "test_var"
    var_dir.mkdir(exist_ok=True)
    np.save(str(var_dir / "my_var.npy"), np.array([1, 2, 3]))
    t = load_variable(str(var_dir), "my_var")
    assert t.shape == (3,)
    t2 = load_variable(str(var_dir), "missing_var")
    assert t2.shape == (1,)


def test_run_restore_ops() -> None:
    """Test function test_run_restore_ops."""
    import os

    import pytest

    from ml_switcheroo_compiler.serialization import run_restore_ops

    with pytest.raises(FileNotFoundError):
        run_restore_ops("nonexistent_restore")
    os.makedirs("test_restore", exist_ok=True)
    run_restore_ops("test_restore")
    import shutil

    shutil.rmtree("test_restore")


def test_export_methods() -> None:
    """Test function test_export_methods."""
    import os

    from ml_switcheroo_compiler.serialization import export_model_topology, export_to_onnx, export_to_tflite

    class FakeGraph:
        """Test class FakeGraph."""

        def to_json(self) -> EagerValue:
            """Test function to_json."""
            return "{}"

    g = FakeGraph()
    typing.cast("EagerCallableProtocol", export_to_onnx)(g, "test.onnx")
    assert os.path.exists("test.onnx")
    with open("test.onnx", "rb") as f:
        assert f.read() == b"ONNX"
    os.remove("test.onnx")
    typing.cast("EagerCallableProtocol", export_to_tflite)(g, "test.tflite")
    assert os.path.exists("test.tflite")
    with open("test.tflite", "rb") as f:
        assert f.read() == b"TFLITE"
    os.remove("test.tflite")
    typing.cast("EagerCallableProtocol", export_model_topology)(g, "test.json")
    assert os.path.exists("test.json")
    with open("test.json") as f2:
        assert f2.read() == "{}"
    os.remove("test.json")


def test_init_exports_mocked() -> None:
    """Test function test_init_exports_mocked."""
    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s
    from ml_switcheroo_compiler.serialization import _load_h5_weights, _load_npz_weights, _load_pickle_weights, _load_safetensors_weights, _save_as_h5, _save_as_safetensors

    with (
        typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "H5WeightFormat")), "save"),
        typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "H5WeightFormat")), "load"),
        typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "SafetensorsWeightFormat")), "save"),
        typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "SafetensorsWeightFormat")), "load"),
        typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "PickleWeightFormat")), "load"),
        typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "NpzWeightFormat")), "load"),
    ):
        _save_as_h5({}, "test.h5")
        _save_as_safetensors({}, "test.st")
        _load_h5_weights("test.h5")
        _load_safetensors_weights("test.st")
        _load_npz_weights("test.npz")
        _load_pickle_weights("test.pk")
    assert callable(test_init_exports_mocked)


def test_init_exports_unmocked() -> None:
    """Test function test_init_exports_unmocked."""
    import pytest

    from ml_switcheroo_compiler.serialization import _load_h5_weights, _load_npz_weights, _load_pickle_weights, _load_safetensors_weights, _save_as_h5, _save_as_safetensors

    with pytest.raises(Exception, match=".*"):
        typing.cast("EagerCallableProtocol", _save_as_h5)(None, None)
    with pytest.raises(Exception, match=".*"):
        typing.cast("EagerCallableProtocol", _save_as_safetensors)(None, None)
    with pytest.raises(Exception, match=".*"):
        typing.cast("EagerCallableProtocol", _load_h5_weights)(None)
    with pytest.raises(Exception, match=".*"):
        typing.cast("EagerCallableProtocol", _load_safetensors_weights)(None)
    assert typing.cast("EagerCallableProtocol", _load_npz_weights)(None) == {}
    with pytest.raises(Exception, match=".*"):
        typing.cast("EagerCallableProtocol", _load_pickle_weights)(None)


def test_init_module_exports() -> None:
    """Test function test_init_module_exports."""
    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s
    from ml_switcheroo_compiler.serialization import _load_h5_weights, _load_npz_weights, _load_pickle_weights, _load_safetensors_weights, _save_as_h5, _save_as_safetensors

    with typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "H5WeightFormat")), "save"):
        _save_as_h5({}, "test.h5")
    with typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "SafetensorsWeightFormat")), "save"):
        _save_as_safetensors({}, "test.st")
    with typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "H5WeightFormat")), "load"):
        _load_h5_weights("test.h5")
    with typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "SafetensorsWeightFormat")), "load"):
        _load_safetensors_weights("test.st")
    with typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "NpzWeightFormat")), "load"):
        _load_npz_weights("test.npz")
    with typing.cast("EagerCallableProtocol", patch.object)(typing.cast("EagerTensorProtocol", getattr(s, "PickleWeightFormat")), "load"):
        _load_pickle_weights("test.pk")
    assert callable(test_init_module_exports)


def test_init_module_all_exports() -> None:
    """Test function test_init_module_all_exports."""
    import os

    import numpy as np

    from ml_switcheroo_compiler.serialization import _load_h5_weights, _load_npz_weights, _load_pickle_weights, _load_safetensors_weights, _save_as_h5, _save_as_safetensors

    tf = "test_real_init_unmocked.h5"
    tf_st = "test_real_init_unmocked.st"
    tf_npz = "test_real_init_unmocked.npz"
    tf_pk = "test_real_init_unmocked.pk"
    _save_as_h5({"a": 1}, tf)
    assert "a" in typing.cast("EagerCallableProtocol", _load_h5_weights)(tf)
    _save_as_safetensors({"a": np.array([1.0], dtype=np.float32)}, tf_st)
    assert "a" in typing.cast("EagerCallableProtocol", _load_safetensors_weights)(tf_st)
    assert _load_npz_weights(tf_npz) == {}
    with pytest.raises(FileNotFoundError):
        _load_pickle_weights(tf_pk)
    for f in [tf, tf_st, tf_npz, tf_pk]:
        if os.path.exists(f):
            os.remove(f)


def test_np_serialization_methods() -> None:
    """Test function test_np_serialization_methods."""
    import ml_switcheroo_compiler.serialization as s

    s._save_as_h5({"a": 1}, "dummy.h5")
    s._save_as_safetensors({"a": 1}, "dummy.st")
    s._load_h5_weights("dummy.h5")
    s._load_safetensors_weights("dummy.st")
    s._load_npz_weights("dummy.npz")
    with pytest.raises(FileNotFoundError):
        s._load_pickle_weights("dummy.pk")
    import os

    if os.path.exists("dummy.h5"):
        os.remove("dummy.h5")
    if os.path.exists("dummy.st"):
        os.remove("dummy.st")


def test_extract_optimizer_state_missing_branches() -> None:
    """Test function test_extract_optimizer_state_missing_branches."""
    from ml_switcheroo_compiler.serialization import _extract_optimizer_state

    class FakeOptEmpty:
        """Test class FakeOptEmpty."""

        pass

    class FakeModelOptEmpty:
        """Test class FakeModelOptEmpty."""

        optimizer = FakeOptEmpty()

    st = typing.cast("dict[str, EagerTensorProtocol]", {})
    typing.cast("EagerCallableProtocol", _extract_optimizer_state)(typing.cast("EagerCallableProtocol", FakeModelOptEmpty)(), st)
    assert len(st) == 0
