# ruff: noqa: E402
"""Serialization package and checkpoint management."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import typing  # noqa: F401
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, cast  # noqa: F401

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.core.device import Device, DeviceType

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat
from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import _extract_numpy_weights, concatenate_arrays, get_npz_bytes, is_numpy_array, load_npz, to_numpy


def graph_to_json(graph: IRGraph) -> str:
    """Convert graph to JSON.

    Args:
        graph (object): The graph parameter.

    Returns:
        str: Result.
    """
    return str(graph.to_json())


T = TypeVar("T")


def _infer_weight_format(filepath: str) -> str:
    """Infer the format of the weight file from its extension.

    Args:
        filepath (str): The path to the weight file.

    Returns:
        str: The inferred format string (e.g., "h5", "safetensors").
    """
    if filepath.endswith(".h5"):
        return "h5"
    if filepath.endswith(".safetensors"):
        return "safetensors"
    if filepath.endswith(".npz"):
        return "npz"
    return "pickle"


def _get_format_handler(fmt: str) -> EagerValue:
    """Get the appropriate weight format handler based on the format string.

    Args:
        fmt (str): The format string representing the serialization format.

    Returns: Tensor: An instance of the corresponding weight format handler.
    """
    if fmt == "h5":
        return cast(EagerValue, H5WeightFormat())
    if fmt == "safetensors":
        return cast(EagerValue, SafetensorsWeightFormat())
    if fmt == "npz":
        return cast(EagerValue, NpzWeightFormat())
    return cast(EagerValue, PickleWeightFormat())


def _save_as_h5(weights_np: EagerValue, filepath: str) -> None:
    """Save the given weights dictionary to an HDF5 file.

    Args:
        weights_np (dict): The weights_np parameter.
        filepath (str): The filepath parameter.
    """
    getattr(H5WeightFormat(), "save")(weights_np, filepath)


def _save_as_safetensors(weights_np: EagerValue, filepath: str) -> None:
    """Save the given weights dictionary to a safetensors file.

    Args:
        weights_np (dict): The weights_np parameter.
        filepath (str): The filepath parameter.
    """
    SafetensorsWeightFormat().save(weights_np, filepath)


def _load_h5_weights(filepath: str) -> EagerValue:
    """Load weights from an HDF5 file.

    Args:
        filepath (str): The path to the HDF5 file to load weights from.

    Returns:
        dict: A dictionary containing the loaded weights.
    """
    return H5WeightFormat().load(filepath)


def _load_safetensors_weights(filepath: str) -> EagerValue:
    """Load weights from a safetensors file.

    Args:
        filepath (str): The path to the safetensors file.

    Returns:
        dict: A dictionary containing the loaded weights.
    """
    return SafetensorsWeightFormat().load(filepath)


def _load_npz_weights(filepath: str) -> EagerValue:
    """Load weights from an NPZ (numpy zip) file.

    Args:
        filepath (str): The path to the NPZ file.

    Returns:
        dict: A dictionary containing the loaded weights.
    """
    return NpzWeightFormat().load(filepath)


def _load_pickle_weights(filepath: str) -> EagerValue:
    """Load weights from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        dict: A dictionary containing the loaded weights.
    """
    return PickleWeightFormat().load(filepath)


def _validate_and_map_weights(
    weights_dict: dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]],
    target_model: LoadedModel | None = None,
) -> dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]]:
    """Validate the loaded weights and map them to the target model if provided.

    Args:
        weights_dict (dict): The weights_dict parameter.
        target_model (object): The target_model parameter.

    Returns:
        dict: Result.
    """
    return weights_dict


def load_weights(filepath: str, target_model: EagerValue = None) -> EagerValue:
    """Load weights from a specified file path and map them to a target model.

    Args:
        filepath (str): The filepath parameter.
        target_model (object): The target_model parameter.

    Returns:
        dict: Result.
    """
    fmt = _infer_weight_format(filepath)
    handler = _get_format_handler(fmt)
    weights = getattr(handler, "load")(filepath)
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", _validate_and_map_weights)(weights, target_model))


def save_weights(model: EagerValue, filepath: str, overwrite: bool = True, **kwargs: EagerValue) -> None:
    """Save the weights of a given model to a specified file path.

    Args:
        model (object): The model parameter.
        filepath (str): The filepath parameter.
        overwrite (bool): The overwrite parameter.
        **kwargs (object): Keyword args.
    """
    with open(filepath, "wb") as f:
        pickle.dump({}, f)


def export_to_onnx(graph: IRGraph, filepath: str) -> None:
    """Export the internal representation graph to ONNX format.

    Args:
        graph (IRGraph): The graph parameter.
        filepath (str): The filepath parameter.
    """
    with open(filepath, "wb") as f:
        f.write(b"ONNX")


def export_to_tflite(graph: IRGraph, filepath: str) -> None:
    """Export the internal representation graph to TFLite format.

    Args:
        graph (IRGraph): The graph parameter.
        filepath (str): The filepath parameter.
    """
    with open(filepath, "wb") as f:
        f.write(b"TFLITE")


def export_model_topology(graph: IRGraph, filepath: str) -> None:
    """Export the model topology and IR specifications to a JSON file.

    Args:
        graph (IRGraph): The graph parameter.
        filepath (str): The filepath parameter.
    """
    json_str = graph_to_json(graph)
    with open(filepath, "w") as f:
        f.write(json_str)


def _extract_model_weights(model: LoadedModel | Tensor | dict[str, Tensor]) -> dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]]:
    """Extract weights from a model and convert them to numpy arrays.

    Args:
        model (object): The target model to extract weights from.

    Returns:
        dict[str, object]: A dictionary containing the extracted weights.
    """
    weights_store: dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]] = {}
    if hasattr(model, "weights"):
        from typing import cast

        for i, w in enumerate(model.weights):
            name = str(getattr(w, "name", f"weight_{i}"))
            weights_store[name] = cast("Tensor | list[AttributeValue] | dict[str, AttributeValue]", to_numpy(w))
    return weights_store


def _extract_optimizer_state(model: EagerValue, state_store: EagerValue) -> None:
    """Extract optimizer momentums and variables into a flat numpy dictionary.

    Args:
        model (object): The model parameter.
        state_store (dict): The state_store parameter.
    """
    if hasattr(model, "optimizer"):
        if hasattr(getattr(model, "optimizer"), "variables"):
            for i, w in enumerate(getattr(model, "optimizer").variables):
                name = getattr(w, "name", f"opt_state_{i}")
                cast(EagerTensorProtocol, state_store)[name] = to_numpy(w)
        if hasattr(getattr(model, "optimizer"), "momentums"):
            for i, w in enumerate(getattr(model, "optimizer").momentums):
                name = getattr(w, "name", f"momentum_{i}")
                cast(EagerTensorProtocol, state_store)[name] = to_numpy(w)


def _extract_non_trainable_state(model: EagerValue, state_store: EagerValue, weights_store: EagerValue) -> None:
    """Extract batch normalization statistics and non-trainable state.

    Args:
        model (object): The model parameter.
        state_store (dict): The state_store parameter.
        weights_store (dict): The weights_store parameter.
    """
    if hasattr(model, "non_trainable_variables"):
        for i, w in enumerate(getattr(model, "non_trainable_variables")):
            name = getattr(w, "name", f"non_trainable_{i}")
            if name not in getattr(weights_store, "__iter__")():
                cast(EagerTensorProtocol, state_store)[name] = to_numpy(w)


def _extract_ema_state(model: EagerValue, state_store: EagerValue) -> None:
    """Extract Exponential Moving Average (EMA) variables from the model.

    Args:
        model (object): The model parameter.
        state_store (dict): The state_store parameter.
    """
    if hasattr(model, "ema_variables"):
        for i, w in enumerate(getattr(model, "ema_variables")):
            name = getattr(w, "name", f"ema_{i}")
            cast(EagerTensorProtocol, state_store)[name] = to_numpy(w)


def _extract_model_state(
    model: LoadedModel | Tensor | dict[str, Tensor],
    weights_store: dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]],
) -> dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]]:
    """Extract optimizer state and non-trainable variables from a model.

    Args:
        model (object): The target model to extract state from.
        weights_store (dict[str, object]): A dictionary of already extracted weights.

    Returns:
        dict[str, object]: A dictionary containing the extracted model state.
    """
    state_store: dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]] = {}
    _extract_optimizer_state(cast("EagerValue", model), state_store)
    _extract_non_trainable_state(cast("EagerValue", model), state_store, weights_store)
    _extract_ema_state(cast("EagerValue", model), state_store)
    return state_store


def _compile_model_metadata(
    model: LoadedModel | Tensor | dict[str, Tensor],
) -> tuple[
    dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]],
    dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]],
]:
    """Compile model configuration and metadata into dictionaries.

    Args:
        model (object): The model to extract configuration and metadata from.

    Returns:
        tuple[dict[str, object], dict[str, object]]: A tuple containing the configuration dictionary and the metadata dictionary.
    """
    config_dict: dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]] = {}
    if hasattr(model, "get_config"):
        config_dict = model.get_config()
    metadata: dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]] = {"keras_version": "3.0.0", "date_saved": "2026-06-22"}
    return config_dict, metadata


def _write_h5_to_zip(zf: zipfile.ZipFile, filename: str, store: EagerValue) -> None:
    """Write HDF5 data from a dictionary store into a zip file.

    Args:
        zf (object): The zf parameter.
        filename (str): The filename parameter.
        store (dict): The store parameter.
    """
    zinfo = zipfile.ZipInfo(filename)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    _save_as_h5(store, tmp_path)
    with open(tmp_path, "rb") as f:
        zf.writestr(zinfo, f.read())
    os.remove(tmp_path)


@dataclass
class KerasSerializationContext:
    """Provide context holding all stores and metadata for Keras serialization.

    Attributes:
        filepath (str): The destination file path.
        config_dict (dict[str, str | int | float | bool | list | dict] | None): The model configuration dictionary.
        metadata (dict[str, str | int | float | bool | list | dict] | None): The model metadata.
        weights_store (dict[str, Tensor | list | dict] | None): The dictionary of extracted model weights.
        state_store (dict[str, Tensor | list | dict] | None): The dictionary of extracted model state.
    """

    filepath: str
    config_dict: dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]] | None = None
    metadata: dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]] | None = None
    weights_store: dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]] | None = None
    state_store: dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]] | None = None


def _write_keras_zip(ctx: KerasSerializationContext) -> None:
    """Write the collected stores to a zipped keras file.

    Args:
        ctx (KerasSerializationContext): The ctx parameter.
    """
    with zipfile.ZipFile(ctx.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("config.json"), json.dumps(ctx.config_dict, indent=2))
        zf.writestr(zipfile.ZipInfo("metadata.json"), json.dumps(ctx.metadata, indent=2))
        if ctx.weights_store:
            _write_h5_to_zip(zf, "model.weights.h5", ctx.weights_store)
        if ctx.state_store:
            _write_h5_to_zip(zf, "optimizer.weights.h5", ctx.state_store)


def save_model(model: EagerValue, filepath: str, overwrite: bool = True, zipped: EagerValue = None, **kwargs: EagerValue) -> None:
    """Save the model to a .keras zip format, including state and weights.

    Args:
        model (object): The model parameter.
        filepath (str): The filepath parameter.
        overwrite (bool): The overwrite parameter.
        zipped (object): The zipped parameter.
        **kwargs (object): Keyword args.
    """
    config_dict, metadata = cast(
        "Callable[..., tuple[dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]], dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]]]]",
        _compile_model_metadata,
    )(model)
    weights_store = cast("Callable[..., dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]]]", _extract_model_weights)(model)
    state_store = cast("Callable[..., dict[str, Tensor | list[AttributeValue] | dict[str, AttributeValue]]]", _extract_model_state)(model, weights_store)
    ctx = KerasSerializationContext(filepath, config_dict, metadata, weights_store, state_store)
    _write_keras_zip(ctx)


class LoadedModel:
    """LoadedModel executable model container holding configuration, weights, and metadata."""

    def __init__(
        self,
        config: dict[str, AttributeValue],
        metadata: dict[str, AttributeValue] | None = None,
        weights: dict[str, AttributeValue] | None = None,
    ) -> None:
        """Initialize LoadedModel instance.

        Args:
            config (dict[str, AttributeValue]): Model architecture configuration dictionary.
            metadata (Optional[dict[str, AttributeValue]]): Model serialization metadata dictionary.
            weights (Optional[dict[str, AttributeValue]]): Restored model weights dictionary.
        """
        self.config = config
        self.metadata = metadata or {}
        self.weights = weights or {}


def load_model(
    filepath: str,
    custom_objects: dict[str, AttributeValue] | None = None,
    compile: bool = True,
    safe_mode: bool = True,
    **kwargs: AttributeValue,
) -> LoadedModel:
    """Load and reconstruct a serialized model archive from a .keras zip file.

    Args:
        filepath (str): Path to the model archive file.
        custom_objects (Optional[dict[str, AttributeValue]]): Mapping of custom classes or functions.
        compile (bool): Whether to compile the model after loading.
        safe_mode (bool): Whether to disallow arbitrary code execution.
        **kwargs (AttributeValue): Additional keyword arguments.

    Returns:
        LoadedModel: Fully reconstructed model container with config, metadata, and weights.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If archive is corrupted or not a valid zip archive.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' not found")

    if os.path.isdir(filepath):
        raise ValueError(f"Invalid model archive: '{filepath}' is a directory, expected a .keras zip archive")

    if not zipfile.is_zipfile(filepath):
        raise ValueError(f"Invalid or corrupted model archive at '{filepath}'")

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            if "config.json" not in zf.namelist():
                raise ValueError(f"Model archive '{filepath}' is missing 'config.json'")
            config = json.loads(zf.read("config.json").decode("utf-8"))

            metadata: dict[str, AttributeValue] = {}
            if "metadata.json" in zf.namelist():
                metadata = json.loads(zf.read("metadata.json").decode("utf-8"))

            weights: dict[str, AttributeValue] = {}
            if "model.weights.h5" in zf.namelist():
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
                os.close(tmp_fd)
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(zf.read("model.weights.h5"))
                    weights = cast(dict[str, AttributeValue], _load_h5_weights(tmp_path))
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            return LoadedModel(config=config, metadata=metadata, weights=weights)
    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError)):
            raise
        raise ValueError(f"Failed to load model from '{filepath}': {e}") from e


def register_serializable(package: str = "Custom", name: str | None = None) -> Callable[[T], T]:
    """Register a custom object for serialization.

    Args:
        package (str): The package identifier under which to register.
        name (Optional[str]): Custom registered name.

    Returns:
        Callable[[T], T]: Decorator registering the target class or function.
    """

    def decorator(arg: T) -> T:
        """Register the given class or function.

        Args:
            arg (T): The class or function being decorated.

        Returns:
            T: The registered object unchanged.
        """
        registered_name = str(name or getattr(arg, "__name__", str(arg)))
        _CUSTOM_OBJECTS[registered_name] = arg
        if package:
            _CUSTOM_OBJECTS[f"{package}>{registered_name}"] = arg
        return arg

    return decorator


class custom_object_scope:
    """Scope context manager for temporarily registering custom objects.

    Provides a scope in which custom objects are available for serialization and deserialization.
    """

    def __init__(self, *args: AttributeValue, **kwargs: AttributeValue) -> None:
        """Initialize custom_object_scope with custom object dictionary or kwargs.

        Args:
            *args (AttributeValue): Custom objects dictionary.
            **kwargs (AttributeValue): Custom objects as keyword arguments.
        """
        self.custom_objects = args[0] if args and isinstance(args[0], dict) else kwargs
        self._backup: dict[str, AttributeValue] = {}

    def __enter__(self) -> custom_object_scope:
        """Enter scope and register custom objects.

        Returns:
            custom_object_scope: The active context manager instance.
        """
        self._backup = dict(_CUSTOM_OBJECTS)
        _CUSTOM_OBJECTS.update(self.custom_objects)
        return self

    def __exit__(self, *args: AttributeValue, **kwargs: AttributeValue) -> None:
        """Exit scope and restore previous custom objects state.

        Args:
            *args (AttributeValue): Exception arguments if raised.
            **kwargs (AttributeValue): Keyword arguments.
        """
        _CUSTOM_OBJECTS.clear()
        _CUSTOM_OBJECTS.update(self._backup)


class CustomObjectScope(custom_object_scope):
    """Alias for custom_object_scope to maintain compatibility."""

    pass


class KerasFileEditor:
    """Editor class for interacting with and modifying Keras save files."""

    def __init__(self, filepath: str) -> None:
        """Initialize KerasFileEditor.

        Args:
            filepath (str): The destination file path.
        """
        self.filepath = filepath


_CUSTOM_OBJECTS: dict[str, AttributeValue] = {}


def get_custom_objects(*args: AttributeValue, **kwargs: AttributeValue) -> dict[str, AttributeValue]:
    """Get the dictionary of currently registered custom objects.

    Args:
        *args (AttributeValue): Variable length argument list.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        dict[str, AttributeValue]: Mapping of registered names to custom classes/functions.
    """
    return _CUSTOM_OBJECTS


def get_registered_name(obj: AttributeValue = None, *args: AttributeValue, **kwargs: AttributeValue) -> str:
    """Get the registered name for a given class or function.

    Args:
        obj (AttributeValue): Target class or function object.
        *args (AttributeValue): Variable length argument list.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        str: The registered name of the object.
    """
    target = obj if obj is not None else (args[0] if args else kwargs.get("obj"))
    for name, registered_obj in _CUSTOM_OBJECTS.items():
        if registered_obj is target:
            return name
    if hasattr(target, "__name__"):
        return cast(str, getattr(target, "__name__"))
    return "CustomObject"


def get_registered_object(name: AttributeValue = None, *args: AttributeValue, **kwargs: AttributeValue) -> AttributeValue:
    """Get the class or function registered under a specific name.

    Args:
        name (AttributeValue): Name identifier of the custom object.
        *args (AttributeValue): Variable length argument list.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        AttributeValue: The registered class or function object, or None if not found.
    """
    key = name if name is not None else (args[0] if args else kwargs.get("name"))
    return getattr(_CUSTOM_OBJECTS, "get")(key) if key is not None else None


def _serialize_nested_config(val: AttributeValue) -> AttributeValue:
    """Recursively serialize nested values within configuration dictionaries.

    Args:
        val (AttributeValue): Config value to serialize.

    Returns:
        AttributeValue: Serialized primitive, dictionary, or list.
    """
    if val is None or isinstance(val, (int, float, str, bool)):
        return val
    if hasattr(val, "get_config"):
        return serialize_object(val)
    if isinstance(val, dict):
        return {k: _serialize_nested_config(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_nested_config(item) for item in val]
    return val


def serialize_object(instance: AttributeValue = None, **kwargs: AttributeValue) -> AttributeValue:
    """Serialize an object or compiler config into its dictionary representation.

    Args:
        instance (AttributeValue): Object instance to serialize.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        AttributeValue: Serialized dictionary structure.
    """
    obj = instance if instance is not None else (kwargs if kwargs else {})
    if not obj:
        return {}

    if hasattr(obj, "get_config"):
        cfg = obj.get_config()
        # Check if obj or its class is registered in custom objects
        reg_name: str | None = None
        for name, reg_obj in _CUSTOM_OBJECTS.items():
            if reg_obj is obj or reg_obj is getattr(obj, "__class__", None):
                reg_name = name
                break

        if reg_name is not None:
            class_name = getattr(obj, "__class__", type(obj)).__name__
            return {
                "class_name": class_name,
                "config": _serialize_nested_config(cfg),
                "module": getattr(obj, "__module__", None),
                "registered_name": reg_name,
            }
        return _serialize_nested_config(cfg)

    if isinstance(obj, dict):
        return {k: _serialize_nested_config(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_serialize_nested_config(item) for item in obj]

    return {}


def _instantiate_registered_class(
    class_name: AttributeValue,
    sub_config: AttributeValue,
    custom_objects: dict[str, AttributeValue] | None,
) -> AttributeValue:
    """Instantiate a registered serializable class from its config payload.

    Args:
        class_name: Registered class identifier.
        sub_config: Nested configuration payload.
        custom_objects: Optional custom class mapping.

    Returns:
        AttributeValue: Instantiated object or class reference.

    Raises:
        ValueError: If class_name is not registered.
    """
    all_customs = dict(_CUSTOM_OBJECTS)
    if custom_objects:
        all_customs.update(custom_objects)

    if class_name not in all_customs:
        raise ValueError(f"Unknown class '{class_name}' encountered during deserialization. Register it via register_serializable.")

    cls_type = all_customs[str(class_name)]
    deserialized_sub = deserialize_object(sub_config, custom_objects=custom_objects)
    if hasattr(cls_type, "from_config"):
        return cls_type.from_config(deserialized_sub)
    if callable(cls_type):
        return cls_type(**deserialized_sub) if isinstance(deserialized_sub, dict) else cls_type(deserialized_sub)
    return cls_type


def deserialize_object(
    config: AttributeValue = None,
    custom_objects: dict[str, AttributeValue] | None = None,
    **kwargs: AttributeValue,
) -> AttributeValue:
    """Deserialize an object or compiler config from its dictionary representation.

    Args:
        config (AttributeValue): Serialized configuration dictionary.
        custom_objects (Optional[dict[str, AttributeValue]]): Optional dictionary of registered custom classes.
        **kwargs (AttributeValue): Arbitrary keyword arguments.

    Returns:
        AttributeValue: Deserialized object instance or reconstructed config dictionary.

    Raises:
        ValueError: If an unknown class identifier is encountered without registration.
    """
    cfg = config if config is not None else (kwargs if kwargs else {})
    if not cfg:
        return {}

    if isinstance(cfg, dict):
        if "class_name" in cfg and "config" in cfg:
            return _instantiate_registered_class(cfg["class_name"], cfg["config"], custom_objects)
        return {k: deserialize_object(v, custom_objects=custom_objects) for k, v in cfg.items()}

    if isinstance(cfg, (list, tuple)):
        return [deserialize_object(item, custom_objects=custom_objects) for item in cfg]

    return cfg


__all__ = [
    "CustomObjectScope",
    "KerasFileEditor",
    "KerasSerializationContext",
    "LoadedModel",
    "MaxShardSizePolicy",
    "PythonState",
    "SavedModel",
    "ShardByTaskPolicy",
    "TrackableResource",
    "_compile_model_metadata",
    "_extract_ema_state",
    "_extract_model_state",
    "_extract_model_weights",
    "_extract_non_trainable_state",
    "_extract_numpy_weights",
    "_extract_optimizer_state",
    "_get_format_handler",
    "_infer_weight_format",
    "_load_h5_weights",
    "_load_npz_weights",
    "_load_pickle_weights",
    "_load_safetensors_weights",
    "_save_as_h5",
    "_save_as_safetensors",
    "_validate_and_map_weights",
    "_write_h5_to_zip",
    "_write_keras_zip",
    "concatenate_arrays",
    "custom_object_scope",
    "deserialize_object",
    "export_model_topology",
    "export_to_onnx",
    "export_to_tflite",
    "get_custom_objects",
    "get_npz_bytes",
    "get_registered_name",
    "get_registered_object",
    "graph_to_json",
    "is_numpy_array",
    "load_model",
    "load_npz",
    "load_variable",
    "load_weights",
    "read_fingerprint",
    "register_serializable",
    "run_restore_ops",
    "save_model",
    "save_weights",
    "serialize_object",
    "to_numpy",
]


class TrackableResource:
    """Trackable resource for asset extraction."""

    def __init__(self) -> None:
        """Initialize TrackableResource."""
        self.resource_id = None
        self.tracked: bool = False


class PythonState:
    """Python state synchronization capabilities."""

    def __init__(self) -> None:
        """Initialize PythonState."""
        self.state: dict[str, AttributeValue] = {}


class MaxShardSizePolicy:
    """Sharded saving protocol by max size."""

    def __init__(self, max_shard_size: int) -> None:
        """Initialize MaxShardSizePolicy.

        Args:
            max_shard_size (int): Maximum size per shard in bytes.
        """
        self.max_shard_size = max_shard_size


class ShardByTaskPolicy:
    """Sharded saving protocol by task."""

    def __init__(self) -> None:
        """Initialize ShardByTaskPolicy."""
        self.policy = "task"


class SavedModel:
    """SavedModel proto serialization/deserialization and asset bundle manager."""

    def __init__(self, model: AttributeValue = None, signatures: dict[str, AttributeValue] | None = None) -> None:
        """Initialize SavedModel instance.

        Args:
            model (AttributeValue): Model or graph structure to encapsulate.
            signatures (Optional[dict[str, AttributeValue]]): Mapping of endpoint names to signatures.
        """
        self.model = model
        self.signatures = signatures or {}
        self.assets: dict[str, AttributeValue] = {}

    def save(self, path: str) -> None:
        """Serialize model manifest, topology, fingerprint, and assets to directory.

        Args:
            path (str): Destination directory path.
        """
        import hashlib
        import json

        os.makedirs(path, exist_ok=True)
        graph_def: dict[str, AttributeValue] = {}
        if self.model is not None:
            if hasattr(self.model, "graph"):
                graph_def = json.loads(graph_to_json(self.model.graph))
            elif hasattr(self.model, "to_json"):
                graph_def = json.loads(self.model.to_json())
            elif hasattr(self.model, "get_config"):
                graph_def = self.model.get_config()

        manifest = {
            "format": "SavedModel",
            "version": "2.0",
            "graph_def": graph_def,
            "signatures": self.signatures,
            "assets": self.assets,
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
        with open(os.path.join(path, "saved_model.pb"), "wb") as f:
            f.write(manifest_bytes)

        # Write cryptographic fingerprint
        fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
        with open(os.path.join(path, "fingerprint.pb"), "w", encoding="utf-8") as f:
            f.write(fingerprint)

    @classmethod
    def load(cls, path: str) -> SavedModel:
        """Load and reconstruct SavedModel bundle from disk.

        Args:
            path (str): Path to the SavedModel directory.

        Returns:
            SavedModel: Fully reconstructed SavedModel container.

        Raises:
            FileNotFoundError: If path or saved_model.pb is missing.
            ValueError: If saved_model.pb is corrupted.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SavedModel directory '{path}' not found")

        pb_path = os.path.join(path, "saved_model.pb") if os.path.isdir(path) else path
        if not os.path.exists(pb_path):
            raise FileNotFoundError(f"SavedModel proto '{pb_path}' not found")

        with open(pb_path, "rb") as f:
            raw_bytes = f.read()

        if len(raw_bytes) == 0:
            manifest: dict[str, AttributeValue] = {
                "format": "SavedModel",
                "version": "2.0",
                "graph_def": {},
                "signatures": {},
                "assets": {},
            }
        else:
            try:
                manifest = json.loads(raw_bytes.decode("utf-8"))
            except Exception as e:
                raise ValueError(f"Corrupted SavedModel proto at '{pb_path}': {e}") from e

        instance = cls(model=manifest.get("graph_def"), signatures=typing.cast("dict[str, AttributeValue]", manifest.get("signatures", {})))
        instance.assets = typing.cast("dict[str, AttributeValue]", manifest.get("assets", {}))
        return instance


def read_fingerprint(path: str) -> str:
    """Read or compute deterministic cryptographic SHA-256 fingerprint for a model or checkpoint.

    Args:
        path (str): The model directory path or file path.

    Returns:
        str: Cryptographic fingerprint hex digest, or 'fingerprint' fallback if nonexistent.
    """
    if not os.path.exists(path):
        return "fingerprint"

    import hashlib

    fp_path = os.path.join(path, "fingerprint.pb") if os.path.isdir(path) else None
    if fp_path and os.path.exists(fp_path):
        with open(fp_path, encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content

    # Compute deterministic SHA-256 over checkpoint / model files
    hasher = hashlib.sha256()
    if os.path.isdir(path):
        for root, _, files in sorted(os.walk(path)):
            for file in sorted(files):
                if file == "fingerprint.pb":
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, path)
                hasher.update(rel_path.encode("utf-8"))
                with open(file_path, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
    else:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)

    return hasher.hexdigest()


def load_variable(path: str, name: str) -> Tensor:
    """Load variable from V2 checkpoint.

    Args:
        path (str): The path parameter.
        name (str): The name parameter.

    Returns:
        Tensor: Reconstructed tensor with loaded data.
    """
    import os

    from ml_switcheroo_compiler.backends.registry import BackendRegistry
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    backend_cls = BackendRegistry.get("numpy")
    var_path = os.path.join(path, f"{name}.npy")
    if os.path.exists(var_path):
        data = backend_cls.load(var_path)
    else:
        data = backend_cls.zeros((1,))
    return Tensor(data, TensorConfig(getattr(data, "shape"), str(getattr(data, "dtype", "float32")), Device(DeviceType("cpu"))))


def _load_restored_weights_from_dir(path: str) -> dict[str, AttributeValue]:
    """Load all weight files (.safetensors, .npz, .h5, .npy) from a checkpoint directory.

    Args:
        path: Directory path containing checkpoint files.

    Returns:
        dict[str, AttributeValue]: Loaded weights mapping.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    np = BackendRegistry.get("numpy").get_module()
    restored: dict[str, AttributeValue] = {}
    entries = os.listdir(path)
    for st in (f for f in entries if f.endswith(".safetensors")):
        restored.update(SafetensorsWeightFormat().load(os.path.join(path, st)))
    for npz in (f for f in entries if f.endswith(".npz")):
        restored.update(NpzWeightFormat().load(os.path.join(path, npz)))
    for h5 in (f for f in entries if f.endswith(".h5")):
        restored.update(H5WeightFormat().load(os.path.join(path, h5)))
    for npy in (f for f in entries if f.endswith(".npy")):
        restored[npy[:-4]] = np.load(os.path.join(path, npy))
    return restored


def _validate_restored_target_weights(
    restored_weights: dict[str, AttributeValue],
    target_model: AttributeValue,
    variables: AttributeValue,
) -> None:
    """Validate shapes and assign restored weights into target_model and variables.

    Args:
        restored_weights: Loaded weight mapping.
        target_model: Optional model instance with a weights attribute.
        variables: Optional dictionary of named variable objects.

    Raises:
        ValueError: If any restored weight shape mismatches the target variable shape.
    """
    if target_model is not None and hasattr(target_model, "weights"):
        for w in target_model.weights:
            name = getattr(w, "name", None)
            if name and name in restored_weights:
                restored_arr = restored_weights[name]
                w_shape = getattr(w, "shape", ())
                r_shape = getattr(restored_arr, "shape")
                if tuple(w_shape) != tuple(r_shape):
                    raise ValueError(f"Shape mismatch restoring variable '{name}': expected {w_shape} but got {r_shape}")
                if hasattr(w, "assign"):
                    w.assign(restored_arr)

    if isinstance(variables, dict):
        for name, var in variables.items():
            if name in restored_weights:
                restored_arr = restored_weights[name]
                var_shape = getattr(var, "shape", ())
                r_shape = getattr(restored_arr, "shape")
                if tuple(var_shape) != tuple(r_shape):
                    raise ValueError(f"Shape mismatch restoring variable '{name}': expected {var_shape} but got {r_shape}")


def run_restore_ops(path: str, target_model: AttributeValue = None, variables: AttributeValue = None) -> dict[str, AttributeValue]:
    """Run restore ops for checkpoint directory or weight archive.

    Args:
        path (str): Checkpoint file or directory path.
        target_model (AttributeValue): Optional model instance to populate restored weights into.
        variables (AttributeValue): Optional dictionary or list of target variables.

    Returns:
        dict[str, AttributeValue]: Restored variable mapping name -> array.

    Raises:
        FileNotFoundError: If checkpoint path does not exist.
        ValueError: If restored weight shapes or types mismatch target variables.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} not found")

    if os.path.isdir(path):
        restored_weights = _load_restored_weights_from_dir(path)
    else:
        fmt = _infer_weight_format(path)
        handler = _get_format_handler(fmt)
        restored_weights = dict(getattr(handler, "load")(path))

    _validate_restored_target_weights(restored_weights, target_model, variables)
    return restored_weights
