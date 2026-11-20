"""Tests for test_model_state_extraction."""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from tests.testing_utils import MockTestStub, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_init_classes_serialization() -> None:
    """Test function test_init_classes_serialization."""
    from ml_switcheroo_compiler.serialization import (
        MaxShardSizePolicy,
        PythonState,
        SavedModel,
        ShardByTaskPolicy,
        TrackableResource,
    )

    class DummyRes(TrackableResource):
        """Test class DummyRes."""

        def initialize(self) -> None:
            """Test function initialize."""
            pass

        def write_to_directory(self, d: BackendArg) -> None:
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

        def deserialize(self, s: BackendArg) -> None:
            """Test function deserialize.

            Args:
                s: Test parameter `s`.
            """
            pass

    class DummyMaxPolicy(MaxShardSizePolicy):
        """Test class DummyMaxPolicy."""

        def apply(self, x: BackendArg) -> None:
            """Test function apply.

            Args:
                x: Test parameter `x`.
            """
            pass

    class DummyShardTask(ShardByTaskPolicy):
        """Test class DummyShardTask."""

        def apply(self, x: BackendArg) -> None:
            """Test function apply.

            Args:
                x: Test parameter `x`.
            """
            pass

    class DummySavedModel(SavedModel):
        """Test class DummySavedModel."""

        def save(self, p: BackendArg) -> None:
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

            Returns:
                SavedModel: Computed output value from `load`.
            """
            del path
            return cls()

    res = DummyRes()
    res.initialize()
    res.write_to_directory("/tmp")
    st = DummyState()
    st.serialize()
    st.deserialize("s")
    DummyMaxPolicy(1024).apply({})
    DummyShardTask().apply({})
    loaded = DummySavedModel.load("/tmp")
    loaded.save("/tmp")
    assert isinstance(loaded, DummySavedModel)


def test_init_other_methods() -> None:
    """Test function test_init_other_methods."""
    from ml_switcheroo_compiler.serialization import (
        ArchiveFileEditor,
        CustomObjectScope,
        _get_format_handler,
        _infer_weight_format,
        custom_object_scope,
        deserialize_object,
        get_registered_name,
        get_registered_object,
        serialize_object,
    )

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
    ArchiveFileEditor("nonexistent")
    serialize_object(list)
    deserialize_object({})


def test_extract_model_state_layers() -> None:
    """Test function test_extract_model_state_layers."""
    from ml_switcheroo_compiler.serialization import (
        _extract_model_weights,
        _extract_non_trainable_state,
        _extract_optimizer_state,
    )

    class FakeW:
        """Test class FakeW."""

        def __init__(self, n: BackendArg) -> None:
            """Test function __init__.

            Args:
                n: Test parameter `n`.
            """
            self.name = n

        def numpy(self) -> BackendArg:
            """Test function numpy.

            Returns:
                BackendArg: Computed output value from `numpy`.
            """
            return 1

    class FakeModel(MockTestStub):
        """Test class FakeModel."""

        weights = [FakeW("w1")]

    ws = _extract_model_weights(FakeModel())
    assert ws["w1"] == 1

    class FakeOpt:
        """Test class FakeOpt."""

        variables = [FakeW("v1")]
        momentums = [FakeW("m1")]

    class FakeModelOpt(MockTestStub):
        """Test class FakeModelOpt."""

        optimizer = FakeOpt()

    st: dict[str, BackendArg] = {}
    _extract_optimizer_state(FakeModelOpt(), st)
    assert st["v1"] == 1
    assert st["m1"] == 1

    class FakeModelNT(MockTestStub):
        """Test class FakeModelNT."""

        non_trainable_variables = [FakeW("nt1")]

    st = {}
    ws = {"nt1": 1}
    _extract_non_trainable_state(FakeModelNT(), st, ws)
    assert "nt1" not in st
    _extract_non_trainable_state(FakeModelNT(), st, {})
    assert st["nt1"] == 1


def test_extract_model_state_submodules() -> None:
    """Test function test_extract_model_state_submodules."""
    from ml_switcheroo_compiler.serialization import (
        ArchiveSerializationContext,
        _compile_model_metadata,
        _extract_model_state,
        _write_archive_zip,
        _write_h5_to_zip,
    )

    class FakeModelNT(MockTestStub):
        """Test class FakeModelNT."""

        def get_config(self) -> BackendArg:
            """Test function get_config.

            Returns:
                BackendArg: Computed output value from `get_config`.
            """
            return {"c": 1}

    st = _extract_model_state(FakeModelNT(), {})
    assert isinstance(st, dict)
    (c, m) = _compile_model_metadata(FakeModelNT())
    assert c == {"c": 1}
    (c2, m2) = _compile_model_metadata(None)
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
    ctx = ArchiveSerializationContext(
        filepath=tf,
        config_dict={},
        metadata={},
        weights_store={"w": 1},
        state_store={"s": 1},
    )
    with patch.object(H5WeightFormat, "save"):
        _write_archive_zip(ctx)
    if os.path.exists(tf):
        os.remove(tf)


def test_extract_model_state_parameters() -> None:
    """Test function test_extract_model_state_parameters."""
    from ml_switcheroo_compiler.serialization import _extract_ema_state

    class FakeW:
        """Test class FakeW."""

        def __init__(self, n: BackendArg) -> None:
            """Test function __init__.

            Args:
                n: Test parameter `n`.
            """
            self.name = n

        def numpy(self) -> BackendArg:
            """Test function numpy.

            Returns:
                BackendArg: Computed output value from `numpy`.
            """
            return 1

    class FakeModelEMA(MockTestStub):
        """Test class FakeModelEMA."""

        ema_variables = [FakeW("ema1"), FakeW("ema2")]

    st: dict[str, BackendArg] = {}
    _extract_ema_state(FakeModelEMA(), st)
    assert st["ema1"] == 1
    assert st["ema2"] == 1


def test_extract_model_state_nested_objects() -> None:
    """Test function test_extract_model_state_nested_objects."""
    import os
    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization.model_archive as ma
    from ml_switcheroo_compiler.serialization import ArchiveSerializationContext, _write_archive_zip

    tf = "test_write2.zip"
    if os.path.exists(tf):
        os.remove(tf)
    ctx1 = ArchiveSerializationContext(filepath=tf, config_dict={}, metadata={}, weights_store={}, state_store={})
    with patch.object(ma, "_write_h5_to_zip") as mock_w:
        _write_archive_zip(ctx1)
        mock_w.assert_not_called()
    if os.path.exists(tf):
        os.remove(tf)
    ctx2 = ArchiveSerializationContext(
        filepath=tf, config_dict={}, metadata={}, weights_store={"w": 1}, state_store={"s": 1}
    )
    with patch.object(ma, "_write_h5_to_zip") as mock_w:
        _write_archive_zip(ctx2)
        assert mock_w.call_count == 2
    if os.path.exists(tf):
        os.remove(tf)


def test_get_registered_name() -> None:
    """Test function test_get_registered_name."""
    from ml_switcheroo_compiler.serialization import get_registered_name

    assert get_registered_name() == "CustomObject"

    class MyC(MockTestStub):
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

    class MyC(MockTestStub):
        """Test class MyC."""

        def get_config(self) -> BackendArg:
            """Test function get_config.

            Returns:
                BackendArg: Computed output value from `get_config`.
            """
            return {"a": 1}

    assert serialize_object() == {}
    assert serialize_object(MyC()) == {"a": 1}
    assert serialize_object(1) == {}


def test_read_fingerprint(tmp_path: Path) -> None:
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


def test_load_variable(tmp_path: Path) -> None:
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


def test_export_methods(tmp_path: Path) -> None:
    """Test function test_export_methods.

    Args:
        tmp_path: Temporary directory fixture.
    """
    import os

    from ml_switcheroo_compiler.serialization import export_model_topology, export_to_onnx, export_to_tflite

    class FakeGraph(MockTestStub):
        """Test class FakeGraph."""

        def to_json(self, indent: int | None = 2) -> str:
            """Test function to_json.

            Args:
                indent: JSON indentation spaces.

            Returns:
                str: Computed output value from `to_json`.
            """
            del indent
            return "{}"

    g = FakeGraph()
    onnx_p = str(tmp_path / "test.onnx")
    tflite_p = str(tmp_path / "test.tflite")
    json_p = str(tmp_path / "test.json")
    real_onnx_p = str(tmp_path / "test_real.onnx")
    real_tflite_p = str(tmp_path / "test_real.tflite")
    require_callable(export_to_onnx)(g, onnx_p)
    assert os.path.exists(onnx_p)
    with open(onnx_p, "rb") as f:
        assert f.read() == b"ONNX"
    require_callable(export_to_tflite)(g, tflite_p)
    assert os.path.exists(tflite_p)
    with open(tflite_p, "rb") as f:
        assert f.read() == b"TFLITE"
    require_callable(export_model_topology)(g, json_p)
    assert os.path.exists(json_p)
    with open(json_p) as f2:
        assert f2.read() == "{}"

    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.serialization import _validate_and_map_weights

    real_g = IRGraph(name="real_export")
    real_g.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(2,))
    real_g.outputs = ["x"]
    export_to_onnx(real_g, real_onnx_p)
    assert os.path.exists(real_onnx_p)
    export_to_tflite(real_g, real_tflite_p)
    assert os.path.exists(real_tflite_p)
    with open(real_tflite_p, "rb") as f_tfl:
        assert f_tfl.read().startswith(b"TFL3")

    class DummyTargetModel(MockTestStub):
        """Dummy model for target_model validation."""

        weights: list[int] = []

    assert _validate_and_map_weights({"a": 1}, DummyTargetModel()) == {"a": 1}


def test_init_exports_mocked() -> None:
    """Test function test_init_exports_mocked."""
    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s
    from ml_switcheroo_compiler.serialization import (
        _load_h5_weights,
        _load_npz_weights,
        _load_pickle_weights,
        _load_safetensors_weights,
        _save_as_h5,
        _save_as_safetensors,
    )

    with (
        patch.object(s.H5WeightFormat, "save") as m_h5_save,
        patch.object(s.H5WeightFormat, "load") as m_h5_load,
        patch.object(s.SafetensorsWeightFormat, "save") as m_st_save,
        patch.object(s.SafetensorsWeightFormat, "load") as m_st_load,
        patch.object(s.PickleWeightFormat, "load") as m_pk_load,
        patch.object(s.NpzWeightFormat, "load") as m_npz_load,
    ):
        _save_as_h5({}, "test.h5")
        _save_as_safetensors({}, "test.st")
        _load_h5_weights("test.h5")
        _load_safetensors_weights("test.st")
        _load_npz_weights("test.npz")
        _load_pickle_weights("test.pk")
        assert m_h5_save.called and m_h5_load.called
        assert m_st_save.called and m_st_load.called
        assert m_pk_load.called and m_npz_load.called


def test_init_exports_unmocked() -> None:
    """Test function test_init_exports_unmocked."""
    import pytest

    from ml_switcheroo_compiler.serialization import (
        _load_h5_weights,
        _load_npz_weights,
        _load_pickle_weights,
        _load_safetensors_weights,
        _save_as_h5,
        _save_as_safetensors,
    )

    with pytest.raises(Exception, match=".*"):
        require_callable(_save_as_h5)(None, None)
    with pytest.raises(Exception, match=".*"):
        require_callable(_save_as_safetensors)(None, None)
    with pytest.raises(Exception, match=".*"):
        require_callable(_load_h5_weights)(None)
    with pytest.raises(Exception, match=".*"):
        require_callable(_load_safetensors_weights)(None)
    assert require_callable(_load_npz_weights)(None) == {}
    with pytest.raises(Exception, match=".*"):
        require_callable(_load_pickle_weights)(None)


def test_init_module_exports() -> None:
    """Test function test_init_module_exports."""
    from unittest.mock import patch

    import ml_switcheroo_compiler.serialization as s
    from ml_switcheroo_compiler.serialization import (
        _load_h5_weights,
        _load_npz_weights,
        _load_pickle_weights,
        _load_safetensors_weights,
        _save_as_h5,
        _save_as_safetensors,
    )

    with patch.object(s.H5WeightFormat, "save") as m1:
        _save_as_h5({}, "test.h5")
        assert m1.called
    with patch.object(s.SafetensorsWeightFormat, "save") as m2:
        _save_as_safetensors({}, "test.st")
        assert m2.called
    with patch.object(s.H5WeightFormat, "load") as m3:
        _load_h5_weights("test.h5")
        assert m3.called
    with patch.object(s.SafetensorsWeightFormat, "load") as m4:
        _load_safetensors_weights("test.st")
        assert m4.called
    with patch.object(s.NpzWeightFormat, "load") as m5:
        _load_npz_weights("test.npz")
        assert m5.called
    with patch.object(s.PickleWeightFormat, "load") as m6:
        _load_pickle_weights("test.pk")
        assert m6.called


def test_init_module_all_exports() -> None:
    """Test function test_init_module_all_exports."""
    import os

    import numpy as np

    from ml_switcheroo_compiler.serialization import (
        _load_h5_weights,
        _load_npz_weights,
        _load_pickle_weights,
        _load_safetensors_weights,
        _save_as_h5,
        _save_as_safetensors,
    )

    tf = "test_real_init_unmocked.h5"
    tf_st = "test_real_init_unmocked.st"
    tf_npz = "test_real_init_unmocked.npz"
    tf_pk = "test_real_init_unmocked.pk"
    _save_as_h5({"a": 1}, tf)
    assert "a" in str(_load_h5_weights(tf))
    _save_as_safetensors({"a": np.array([1.0], dtype=np.float32)}, tf_st)
    assert "a" in str(_load_safetensors_weights(tf_st))
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

    class FakeModelOptEmpty(MockTestStub):
        """Test class FakeModelOptEmpty."""

        optimizer = FakeOptEmpty()

    st: dict[str, BackendArg] = {}
    _extract_optimizer_state(FakeModelOptEmpty(), st)
    assert len(st) == 0
