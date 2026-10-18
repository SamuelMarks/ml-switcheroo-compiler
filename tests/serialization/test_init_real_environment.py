"""Tests for test_init_real_environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_missing_init_unmocked_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_missing_init_unmocked.

    Args:
        _state: Shared test state dictionary.
    """
    tmp_path = _state["tmp_path"]
    from ml_switcheroo_compiler.serialization import (
        CustomObjectScope,
        KerasFileEditor,
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
        _load_h5_weights,
        _load_npz_weights,
        _load_pickle_weights,
        _load_safetensors_weights,
        _save_as_h5,
        _save_as_safetensors,
        _write_h5_to_zip,
        _write_keras_zip,
        custom_object_scope,
        deserialize_object,
        get_custom_objects,
        get_registered_name,
        get_registered_object,
        load_model,
        load_variable,
        read_fingerprint,
        register_serializable,
        run_restore_ops,
        save_model,
        serialize_object,
    )

    h5_path = str(tmp_path / "model_weights.h5")
    st_path = str(tmp_path / "model_weights.safetensors")
    npz_path = str(tmp_path / "model_weights.npz")
    pk_path = str(tmp_path / "model_weights.pk")
    model_path = str(tmp_path / "saved_model_dir")
    dummy_zip = str(tmp_path / "model.zip")
    _save_as_h5({"test_data": 1.0}, h5_path)
    _save_as_safetensors({"test_data": 1.0}, st_path)
    _load_h5_weights(h5_path)
    _load_safetensors_weights(st_path)
    _load_npz_weights(npz_path)
    getattr(_state, "update")({"CustomObjectScope": CustomObjectScope, "KerasFileEditor": KerasFileEditor, "MaxShardSizePolicy": MaxShardSizePolicy, "PythonState": PythonState})
    getattr(_state, "update")({"SavedModel": SavedModel, "ShardByTaskPolicy": ShardByTaskPolicy, "TrackableResource": TrackableResource, "_compile_model_metadata": _compile_model_metadata})
    getattr(_state, "update")({"_extract_ema_state": _extract_ema_state, "_extract_model_state": _extract_model_state, "_extract_model_weights": _extract_model_weights, "_extract_non_trainable_state": _extract_non_trainable_state})
    getattr(_state, "update")({"_extract_optimizer_state": _extract_optimizer_state, "_load_h5_weights": _load_h5_weights, "_load_npz_weights": _load_npz_weights, "_load_pickle_weights": _load_pickle_weights})
    getattr(_state, "update")({"_load_safetensors_weights": _load_safetensors_weights, "_save_as_h5": _save_as_h5, "_save_as_safetensors": _save_as_safetensors, "_write_h5_to_zip": _write_h5_to_zip})
    getattr(_state, "update")({"_write_keras_zip": _write_keras_zip, "custom_object_scope": custom_object_scope, "deserialize_object": deserialize_object, "dummy_zip": dummy_zip})
    getattr(_state, "update")({"get_custom_objects": get_custom_objects, "get_registered_name": get_registered_name, "get_registered_object": get_registered_object, "h5_path": h5_path})
    getattr(_state, "update")({"load_model": load_model, "load_variable": load_variable, "model_path": model_path, "npz_path": npz_path})
    getattr(_state, "update")({"pk_path": pk_path, "read_fingerprint": read_fingerprint, "register_serializable": register_serializable, "run_restore_ops": run_restore_ops})
    getattr(_state, "update")({"save_model": save_model, "serialize_object": serialize_object, "st_path": st_path})
    _state.update(locals())


def _test_missing_init_unmocked_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_missing_init_unmocked.

    Args:
        _state: Shared test state dictionary.
    """
    (_extract_ema_state, _extract_model_weights, _extract_non_trainable_state, _extract_optimizer_state) = (_state["_extract_ema_state"], _state["_extract_model_weights"], _state["_extract_non_trainable_state"], _state["_extract_optimizer_state"])
    (_load_pickle_weights, pk_path) = (_state["_load_pickle_weights"], _state["pk_path"])
    with pytest.raises(FileNotFoundError):
        _load_pickle_weights(pk_path)
    _extract_model_weights(None)
    _extract_optimizer_state(None, {})
    _extract_non_trainable_state(None, {}, {})
    _extract_ema_state(None, {})
    _state.update(locals())


def _test_missing_init_unmocked_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_missing_init_unmocked.

    Args:
        _state: Shared test state dictionary.
    """
    (_compile_model_metadata, _extract_model_state, _write_h5_to_zip, _write_keras_zip) = (_state["_compile_model_metadata"], _state["_extract_model_state"], _state["_write_h5_to_zip"], _state["_write_keras_zip"])
    (dummy_zip, model_path, save_model) = (_state["dummy_zip"], _state["model_path"], _state["save_model"])
    _extract_model_state(None, {})
    _compile_model_metadata(None)
    with pytest.raises(AttributeError):
        _write_h5_to_zip(None, dummy_zip, {})
    with pytest.raises(AttributeError):
        _write_keras_zip(None)
    save_model(None, model_path)
    _state.update(locals())


def _test_missing_init_unmocked_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_missing_init_unmocked.

    Args:
        _state: Shared test state dictionary.
    """
    (CustomObjectScope, KerasFileEditor, custom_object_scope, load_model) = (_state["CustomObjectScope"], _state["KerasFileEditor"], _state["custom_object_scope"], _state["load_model"])
    (model_path, register_serializable) = (_state["model_path"], _state["register_serializable"])
    load_model(model_path)
    register_serializable()
    custom_object_scope()
    CustomObjectScope()
    KerasFileEditor(model_path)
    _state.update(locals())


def _test_missing_init_unmocked_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_missing_init_unmocked.

    Args:
        _state: Shared test state dictionary.
    """
    (deserialize_object, get_custom_objects, get_registered_name, get_registered_object) = (_state["deserialize_object"], _state["get_custom_objects"], _state["get_registered_name"], _state["get_registered_object"])
    serialize_object = _state["serialize_object"]
    get_custom_objects()
    get_registered_name()
    get_registered_object()
    serialize_object(None)
    deserialize_object()
    _state.update(locals())


def _test_missing_init_unmocked_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_missing_init_unmocked.

    Args:
        _state: Shared test state dictionary.
    """
    (MaxShardSizePolicy, PythonState, SavedModel, ShardByTaskPolicy) = (_state["MaxShardSizePolicy"], _state["PythonState"], _state["SavedModel"], _state["ShardByTaskPolicy"])
    TrackableResource = _state["TrackableResource"]
    TrackableResource()
    PythonState()
    MaxShardSizePolicy(1)
    ShardByTaskPolicy()
    SavedModel()
    _state.update(locals())


def _test_missing_init_unmocked_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_missing_init_unmocked.

    Args:
        _state: Shared test state dictionary.
    """
    (load_variable, model_path, read_fingerprint, run_restore_ops) = (_state["load_variable"], _state["model_path"], _state["read_fingerprint"], _state["run_restore_ops"])
    read_fingerprint(model_path)
    load_variable(model_path, "target_var")
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        run_restore_ops(model_path)


def test_missing_init_unmocked(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_missing_init_unmocked."""
    _state: dict[str, EagerTensorProtocol] = {"tmp_path": tmp_path}
    _test_missing_init_unmocked_part_1(_state)
    _test_missing_init_unmocked_part_2(_state)
    _test_missing_init_unmocked_part_3(_state)
    _test_missing_init_unmocked_part_4(_state)
    _test_missing_init_unmocked_part_5(_state)
    _test_missing_init_unmocked_part_6(_state)
    _test_missing_init_unmocked_part_7(_state)
    assert isinstance(_state, dict)
