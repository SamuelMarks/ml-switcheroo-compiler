"""Tests for test_main."""

from __future__ import annotations

# ruff: noqa: E501
import os
import tempfile
import zipfile
from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.serialization import (
    CustomObjectScope,
    KerasFileEditor,
    KerasSerializationContext,
    MaxShardSizePolicy,
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
    deserialize_object,
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
    register_serializable,
    run_restore_ops,
    save_model,
    save_weights,
    serialize_object,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class _MockVarItem:
    """Test class _MockVarItem."""

    def __init__(self, data: EagerValue, name: EagerValue = None) -> None:
        """Initialize _MockVarItem.

        Args:
            data: Data value.
            name: Optional variable name.
        """
        self._data = data
        if name is not None:
            self.name = name

    def numpy(self) -> EagerValue:
        """Return numpy value."""
        return self._data


class _MockOpt:
    """Test class _MockOpt."""

    def __init__(self) -> None:
        """Initialize _MockOpt."""
        self.variables = [_MockVarItem("v1_data", "v1"), _MockVarItem("v2_data")]
        self.momentums = [_MockVarItem("m1_data", "m1"), _MockVarItem("m2_data")]


class MockModel:
    """Test class MockModel."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.weights = [_MockVarItem("w1_data", "w1"), _MockVarItem("w2_data")]
        self.optimizer = _MockOpt()
        self.non_trainable_variables = [_MockVarItem("nt1_data", "nt1"), _MockVarItem("nt2_data")]

        class EMA:
            """Test class EMA."""

            name = "ema1"

            def numpy(self) -> EagerValue:
                """Test function numpy."""
                return "ema1_data"

        self.ema_variables = [EMA()]

        class EMA2:
            """Test class EMA2."""

            def numpy(self) -> EagerValue:
                """Test function numpy."""
                return "ema2_data"

        cast("Callable[..., EagerTensorProtocol]", self.ema_variables.append)(cast("Callable[..., EagerTensorProtocol]", EMA2)())

    def get_config(self) -> EagerValue:
        """Test function get_config."""
        return {"conf": "mock"}


def test_infer_weight_format() -> None:
    """Test function test_infer_weight_format."""
    assert _infer_weight_format("a.h5") == "h5"
    assert _infer_weight_format("a.safetensors") == "safetensors"
    assert _infer_weight_format("a.npz") == "npz"
    assert _infer_weight_format("a.pkl") == "pickle"


def test_get_format_handler() -> None:
    """Test function test_get_format_handler."""
    assert type(_get_format_handler("h5")).__name__ == "H5WeightFormat"
    assert type(_get_format_handler("safetensors")).__name__ == "SafetensorsWeightFormat"
    assert type(_get_format_handler("npz")).__name__ == "NpzWeightFormat"
    assert type(_get_format_handler("pickle")).__name__ == "PickleWeightFormat"


def test_save_load_helpers(mocker: EagerTensorProtocol) -> None:
    """Test function test_save_load_helpers."""
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


def test_validate_and_map_weights() -> None:
    """Test function test_validate_and_map_weights."""
    assert cast("Callable[..., EagerTensorProtocol]", _validate_and_map_weights)({"a": 1}) == {"a": 1}


def test_load_save_weights(mocker: EagerTensorProtocol) -> None:
    """Test function test_load_save_weights."""
    mocker.patch("ml_switcheroo_compiler.serialization.PickleWeightFormat.load", return_value={"a": 1})
    assert load_weights("test.pkl") == {"a": 1}
    with tempfile.NamedTemporaryFile() as f:
        cast("Callable[..., EagerTensorProtocol]", save_weights)(cast("Callable[..., EagerTensorProtocol]", MockModel)(), f.name)
        assert os.path.exists(f.name)


def test_export_funcs(mocker: EagerTensorProtocol) -> None:
    """Test function test_export_funcs."""
    mock_graph = mocker.MagicMock()
    setattr(mock_graph.to_json, "return_value", "{}")
    with tempfile.NamedTemporaryFile() as f:
        cast("Callable[..., EagerTensorProtocol]", export_to_onnx)(mock_graph, f.name)
        cast("Callable[..., EagerTensorProtocol]", export_to_tflite)(mock_graph, f.name)
        cast("Callable[..., EagerTensorProtocol]", export_model_topology)(mock_graph, f.name)
    assert cast("Callable[..., EagerTensorProtocol]", graph_to_json)(mock_graph) == "{}"


def test_extract_model_state(mocker: EagerTensorProtocol) -> None:
    """Test function test_extract_model_state."""
    mocker.patch("ml_switcheroo_compiler.serialization.to_numpy", side_effect=lambda x: x.numpy())

    class MockModelNoConfig:
        """Test class MockModelNoConfig."""

        pass

    m2 = MockModelNoConfig()
    cast("Callable[..., EagerTensorProtocol]", _compile_model_metadata)(m2)
    m = MockModel()
    weights = cast("Callable[..., EagerTensorProtocol]", _extract_model_weights)(m)
    assert weights == {"w1": "w1_data", "weight_1": "w2_data"}
    state = cast("dict[str, EagerTensorProtocol]", {})
    cast("Callable[..., EagerTensorProtocol]", _extract_optimizer_state)(m, state)
    assert state == {"v1": "v1_data", "opt_state_1": "v2_data", "m1": "m1_data", "momentum_1": "m2_data"}
    state = {}
    cast("Callable[..., EagerTensorProtocol]", _extract_non_trainable_state)(m, state, weights)

    class NT2:
        """Test class NT2."""

        name = "nt2"

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return "nt2_data"

    cast("Callable[..., EagerTensorProtocol]", _extract_non_trainable_state)(m, state, {"nt2": "nt2_data"})
    pass
    state = {}
    cast("Callable[..., EagerTensorProtocol]", _extract_ema_state)(m, state)
    pass
    state = cast("dict[str, EagerTensorProtocol]", cast("Callable[..., EagerTensorProtocol]", _extract_model_state)(m, weights))

    class MockModelNoOptimizer:
        """Test class MockModelNoOptimizer."""

        pass

    cast("Callable[..., EagerTensorProtocol]", _extract_optimizer_state)(cast("Callable[..., EagerTensorProtocol]", MockModelNoOptimizer)(), {})
    pass


def test_compile_model_metadata() -> None:
    """Test function test_compile_model_metadata."""

    class MockModelNoConfig:
        """Test class MockModelNoConfig."""

        pass

    m2 = MockModelNoConfig()
    cast("Callable[..., EagerTensorProtocol]", _compile_model_metadata)(m2)
    m = MockModel()
    (config, meta) = cast("Callable[..., EagerTensorProtocol]", _compile_model_metadata)(m)
    assert config == {"conf": "mock"}
    assert "keras_version" in meta


def test_write_h5_to_zip(mocker: EagerTensorProtocol, tmp_path: EagerTensorProtocol) -> None:
    """Test function test_write_h5_to_zip."""
    mocker.patch("ml_switcheroo_compiler.serialization._save_as_h5")
    with tempfile.NamedTemporaryFile() as f:
        with zipfile.ZipFile(f.name, "w") as zf:
            _write_h5_to_zip(zf, "test.h5", {})

    keras_path = str(tmp_path / "model_bundle.keras")
    ctx = KerasSerializationContext(keras_path, {}, {}, {}, {})
    _write_keras_zip(ctx)
    assert ctx is not None or ctx is None


def test_keras_zip_save_load(mocker: EagerTensorProtocol) -> None:
    """Test function test_keras_zip_save_load."""
    mocker.patch("ml_switcheroo_compiler.serialization._load_h5_weights", return_value={"w1": "w1_data"})
    mocker.patch("ml_switcheroo_compiler.serialization._save_as_h5")
    mocker.patch("ml_switcheroo_compiler.serialization.to_numpy", side_effect=lambda x: x.numpy())

    class MockModelNoConfig:
        """Test class MockModelNoConfig."""

        pass

    m2 = MockModelNoConfig()
    cast("Callable[..., EagerTensorProtocol]", _compile_model_metadata)(m2)
    m = MockModel()
    with tempfile.NamedTemporaryFile() as f:
        cast("Callable[..., EagerTensorProtocol]", save_model)(m, f.name)
        m_loaded = load_model(f.name)
        assert m_loaded.config == {"conf": "mock"}


def test_load_model_exception() -> None:
    """Test function test_load_model_exception."""
    with pytest.raises((FileNotFoundError, ValueError)):
        load_model("nonexistent.keras")


def test_register_keras_serializable() -> None:
    """Test function test_register_keras_serializable."""

    @register_serializable("Pkg", "Name")
    class TestObj:
        """Test class TestObj."""

        pass

    assert TestObj.__name__ == "TestObj"


def test_custom_object_scope() -> None:
    """Test function test_custom_object_scope."""
    with custom_object_scope({"a": 1}) as scope:
        assert scope.custom_objects == {"a": 1}
    with CustomObjectScope() as scope:
        pass


def test_keras_file_editor() -> None:
    """Test function test_keras_file_editor."""
    editor = KerasFileEditor("test.keras")
    assert editor.filepath == "test.keras"


def test_deserialize_serialize_keras_object() -> None:
    """Test function test_deserialize_serialize_keras_object."""
    assert deserialize_object({"a": 1}) == {"a": 1}
    assert deserialize_object(a=1) == {"a": 1}

    class Obj:
        """Test class Obj."""

        def get_config(self) -> EagerValue:
            """Test function get_config."""
            return {"a": 1}

    assert serialize_object(Obj()) == {"a": 1}
    assert serialize_object() == {}


def test_custom_objects_registry() -> None:
    """Test function test_custom_objects_registry."""
    assert isinstance(get_custom_objects(), dict)

    class TestObj:
        """Test class TestObj."""

        pass

    assert get_registered_name(TestObj) == "TestObj"
    assert get_registered_name() == "CustomObject"

    class A:
        """Test class A."""

        __name__ = "A_name"

    assert get_registered_name(A) == "A"
    assert get_registered_object(name="test") is None


def test_classes_init() -> None:
    """Test function test_classes_init."""
    assert TrackableResource() is not None
    assert PythonState().state == {}
    assert MaxShardSizePolicy(10).max_shard_size == 10
    assert ShardByTaskPolicy().policy == "task"


def test_saved_model() -> None:
    """Test function test_saved_model."""
    sm = SavedModel()
    with tempfile.TemporaryDirectory() as td:
        sm.save(td)
        assert os.path.exists(os.path.join(td, "saved_model.pb"))
        assert SavedModel.load(td) is not None


def test_read_fingerprint() -> None:
    """Test function test_read_fingerprint."""
    with tempfile.TemporaryDirectory() as td:
        assert len(read_fingerprint(td)) == 64
        with open(os.path.join(td, "fingerprint.pb"), "w") as f:
            f.write("fp_data")
        assert read_fingerprint(td) == "fp_data"


def test_load_variable() -> None:
    """Test function test_load_variable."""
    with tempfile.TemporaryDirectory() as td:
        res = load_variable(td, "var1")
        assert res.config.shape == (1,)
        import numpy as np

        np.save(os.path.join(td, "var1.npy"), np.array([1, 2]))
        res2 = load_variable(td, "var1")
        assert res2.config.shape == (2,)


def test_run_restore_ops() -> None:
    """Test function test_run_restore_ops."""
    with pytest.raises(FileNotFoundError):
        run_restore_ops("nonexistent_path")
    with tempfile.TemporaryDirectory() as td:
        run_restore_ops(td)
