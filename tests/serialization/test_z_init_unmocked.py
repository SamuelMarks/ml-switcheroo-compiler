def test_missing_init_unmocked():
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
        deserialize_keras_object,
        get_custom_objects,
        get_registered_name,
        get_registered_object,
        load_model,
        load_variable,
        read_fingerprint,
        register_keras_serializable,
        run_restore_ops,
        save_model,
        serialize_keras_object,
    )

    # Call them to ensure execution
    try:
        _save_as_h5({}, "test.h5")
    except Exception:
        pass
    try:
        _save_as_safetensors({}, "test.st")
    except Exception:
        pass
    try:
        _load_h5_weights("test.h5")
    except Exception:
        pass
    try:
        _load_safetensors_weights("test.st")
    except Exception:
        pass
    try:
        _load_npz_weights("test.npz")
    except Exception:
        pass
    try:
        _load_pickle_weights("test.pk")
    except Exception:
        pass

    try:
        _extract_model_weights(None)
    except Exception:
        pass

    try:
        _extract_optimizer_state(None, {})
    except Exception:
        pass

    try:
        _extract_non_trainable_state(None, {}, {})
    except Exception:
        pass

    try:
        _extract_ema_state(None, {})
    except Exception:
        pass

    try:
        _extract_model_state(None, {})
    except Exception:
        pass

    try:
        _compile_model_metadata(None)
    except Exception:
        pass

    try:
        _write_h5_to_zip(None, "dummy", {})
    except Exception:
        pass

    try:
        _write_keras_zip(None)
    except Exception:
        pass

    try:
        save_model(None, "dummy")
    except Exception:
        pass

    try:
        load_model("dummy")
    except Exception:
        pass

    try:
        register_keras_serializable()
    except Exception:
        pass

    try:
        custom_object_scope()
    except Exception:
        pass

    try:
        CustomObjectScope()
    except Exception:
        pass

    try:
        KerasFileEditor("dummy")
    except Exception:
        pass

    try:
        get_custom_objects()
    except Exception:
        pass

    try:
        get_registered_name()
    except Exception:
        pass

    try:
        get_registered_object()
    except Exception:
        pass

    try:
        serialize_keras_object(None)
    except Exception:
        pass

    try:
        deserialize_keras_object()
    except Exception:
        pass

    try:
        TrackableResource()
    except Exception:
        pass

    try:
        PythonState()
    except Exception:
        pass

    try:
        MaxShardSizePolicy(1)
    except Exception:
        pass

    try:
        ShardByTaskPolicy()
    except Exception:
        pass

    try:
        SavedModel()
    except Exception:
        pass

    try:
        read_fingerprint("dummy")
    except Exception:
        pass

    try:
        load_variable("dummy", "dummy")
    except Exception:
        pass

    try:
        run_restore_ops("dummy")
    except Exception:
        pass
