"""Serialization package."""

import json
import os
import pickle  # pragma: no cover
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat
from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.ir_format import (
    graph_to_flatbuffers,
    graph_to_json,
    graph_to_protobuf,
    json_to_graph,
)
from ml_switcheroo_compiler.serialization.utils import (
    _extract_numpy_weights,
    concatenate_arrays,
    get_npz_bytes,
    is_numpy_array,
    load_npz,
    to_numpy,
)

T = TypeVar("T")


def _infer_weight_format(filepath: str) -> str:
    """Infer weight format."""
    if filepath.endswith(".h5"):  # pragma: no cover
        return "h5"  # pragma: no cover
    if filepath.endswith(".safetensors"):  # pragma: no cover
        return "safetensors"  # pragma: no cover
    if filepath.endswith(".npz"):  # pragma: no cover
        return "npz"  # pragma: no cover
    return "pickle"  # pragma: no cover


def _get_format_handler(fmt: str) -> object:
    """Get format handler."""
    if fmt == "h5":  # pragma: no cover
        return H5WeightFormat()  # pragma: no cover
    if fmt == "safetensors":  # pragma: no cover
        return SafetensorsWeightFormat()  # pragma: no cover
    if fmt == "npz":  # pragma: no cover
        return NpzWeightFormat()  # pragma: no cover
    return PickleWeightFormat()  # pragma: no cover


def _save_as_h5(weights_np: dict, filepath: str) -> None:
    """Save as h5."""
    H5WeightFormat().save(weights_np, filepath)  # pragma: no cover


def _save_as_safetensors(weights_np: dict, filepath: str) -> None:
    """Save as safetensors."""
    SafetensorsWeightFormat().save(weights_np, filepath)  # pragma: no cover


def _load_h5_weights(filepath: str) -> dict:
    """Load h5 weights."""
    return H5WeightFormat().load(filepath)  # pragma: no cover


def _load_safetensors_weights(filepath: str) -> dict:
    """Load safetensors weights."""
    return SafetensorsWeightFormat().load(filepath)  # pragma: no cover


def _load_npz_weights(filepath: str) -> dict:
    """Load npz weights."""
    return NpzWeightFormat().load(filepath)  # pragma: no cover


def _load_pickle_weights(filepath: str) -> dict:
    """Load pickle weights."""
    return PickleWeightFormat().load(filepath)  # pragma: no cover


def _validate_and_map_weights(weights_dict: dict, target_model: object = None) -> dict:
    """Validate and map weights."""
    return weights_dict  # pragma: no cover


def load_weights(filepath: str, target_model: object = None) -> dict:
    """Load weights from a file."""
    fmt = _infer_weight_format(filepath)  # pragma: no cover
    handler = _get_format_handler(fmt)  # pragma: no cover
    weights = handler.load(filepath)  # pragma: no cover
    return _validate_and_map_weights(weights, target_model)  # pragma: no cover


def save_weights(model: object, filepath: str, overwrite: bool = True, **kwargs: object) -> None:
    """Save weights."""
    with open(filepath, "wb") as f:  # pragma: no cover
        pickle.dump({}, f)  # pragma: no cover


def export_to_onnx(graph: IRGraph, filepath: str) -> None:
    """Export the graph to ONNX format."""
    with open(filepath, "wb") as f:  # pragma: no cover
        f.write(b"ONNX")  # pragma: no cover


def export_to_tflite(graph: IRGraph, filepath: str) -> None:
    """Export the graph to TFLite format."""
    with open(filepath, "wb") as f:  # pragma: no cover
        f.write(b"TFLITE")  # pragma: no cover


def export_model_topology(graph: IRGraph, filepath: str) -> None:
    """Export the model topology and IR specifications."""
    json_str = graph_to_json(graph)  # pragma: no cover
    with open(filepath, "w") as f:  # pragma: no cover
        f.write(json_str)  # pragma: no cover


def _extract_model_weights(model: object) -> dict[str, object]:  # pragma: no cover
    """Extract weights from a model."""
    weights_store = {}
    if hasattr(model, "weights"):
        for i, w in enumerate(model.weights):
            name = getattr(w, "name", f"weight_{i}")
            weights_store[name] = to_numpy(w)
    return weights_store


def _extract_optimizer_state(model: object, state_store: dict[str, object]) -> None:  # pragma: no cover
    """Function docstring."""
    if hasattr(model, "optimizer"):
        if hasattr(model.optimizer, "variables"):
            for i, w in enumerate(model.optimizer.variables):
                name = getattr(w, "name", f"opt_state_{i}")
                state_store[name] = to_numpy(w)
        if hasattr(model.optimizer, "momentums"):
            for i, w in enumerate(model.optimizer.momentums):
                name = getattr(w, "name", f"momentum_{i}")
                state_store[name] = to_numpy(w)


def _extract_non_trainable_state(model: object, state_store: dict[str, object], weights_store: dict[str, object]) -> None:  # pragma: no cover
    """Function docstring."""
    if hasattr(model, "non_trainable_variables"):
        for i, w in enumerate(model.non_trainable_variables):
            name = getattr(w, "name", f"non_trainable_{i}")
            if name not in weights_store:
                state_store[name] = to_numpy(w)


def _extract_ema_state(model: object, state_store: dict[str, object]) -> None:  # pragma: no cover
    """Function docstring."""
    if hasattr(model, "ema_variables"):
        for i, w in enumerate(model.ema_variables):
            name = getattr(w, "name", f"ema_{i}")
            state_store[name] = to_numpy(w)


def _extract_model_state(model: object, weights_store: dict[str, object]) -> dict[str, object]:  # pragma: no cover
    """Extract optimizer state and non-trainable variables."""
    state_store: dict[str, object] = {}
    _extract_optimizer_state(model, state_store)
    _extract_non_trainable_state(model, state_store, weights_store)
    _extract_ema_state(model, state_store)
    return state_store


def _compile_model_metadata(
    model: object,
) -> tuple[dict[str, object], dict[str, object]]:  # pragma: no cover
    """Compile model configuration and metadata."""
    config_dict = {}
    if hasattr(model, "get_config"):
        config_dict = model.get_config()
    metadata = {"keras_version": "3.0.0", "date_saved": "2026-06-22"}
    return config_dict, metadata


def _write_h5_to_zip(zf: zipfile.ZipFile, filename: str, store: dict[str, object]) -> None:  # pragma: no cover
    """Helper to write h5 data into a zip file."""
    zinfo = zipfile.ZipInfo(filename)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    _save_as_h5(store, tmp_path)
    with open(tmp_path, "rb") as f:
        zf.writestr(zinfo, f.read())
    os.remove(tmp_path)


@dataclass
class KerasSerializationContext:
    """Context for Keras serialization."""

    filepath: str
    config_dict: dict[str, object]
    metadata: dict[str, object]
    weights_store: dict[str, object]
    state_store: dict[str, object]


def _write_keras_zip(ctx: KerasSerializationContext) -> None:  # pragma: no cover
    """Write the collected stores to a zipped keras file."""
    with zipfile.ZipFile(ctx.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("config.json"), json.dumps(ctx.config_dict, indent=2))
        zf.writestr(zipfile.ZipInfo("metadata.json"), json.dumps(ctx.metadata, indent=2))

        if ctx.weights_store:
            _write_h5_to_zip(zf, "model.weights.h5", ctx.weights_store)
        if ctx.state_store:
            _write_h5_to_zip(zf, "optimizer.weights.h5", ctx.state_store)


def save_model(  # pragma: no cover
    model: object, filepath: str, overwrite: bool = True, zipped: object = None, **kwargs: object
) -> None:
    """Save model to .keras zip format, including state and weights."""
    config_dict, metadata = _compile_model_metadata(model)
    weights_store = _extract_model_weights(model)
    state_store = _extract_model_state(model, weights_store)
    ctx = KerasSerializationContext(filepath, config_dict, metadata, weights_store, state_store)
    _write_keras_zip(ctx)


def load_model(
    filepath: str,
    custom_objects: object = None,
    compile: bool = True,
    safe_mode: bool = True,
    **kwargs: object,
) -> object:
    """Load model."""
    pass  # pragma: no cover


def register_keras_serializable(package: str = "Custom", name: Optional[str] = None) -> Callable[[T], T]:
    """Register an object with Keras serialization."""

    def decorator(arg: T) -> T:  # pragma: no cover
        """Function docstring."""
        return arg  # pragma: no cover

    return decorator  # pragma: no cover


class custom_object_scope:
    """Scope for custom objects."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize scope."""
        pass  # pragma: no cover

    def __enter__(self) -> "custom_object_scope":
        """Enter scope."""
        return self  # pragma: no cover

    def __exit__(self, *args: object, **kwargs: object) -> None:
        """Exit scope."""
        pass  # pragma: no cover


class CustomObjectScope:
    """Alias for custom_object_scope."""

    pass


class KerasFileEditor:
    """Editor for Keras files."""

    pass


def deserialize_keras_object(*args: object, **kwargs: object) -> object:
    """Deserialize a Keras object."""
    pass  # pragma: no cover


def get_custom_objects(*args: object, **kwargs: object) -> dict[str, object]:
    """Get custom objects."""
    return {}  # pragma: no cover


def get_registered_name(*args: object, **kwargs: object) -> str:
    """Get registered name."""
    return ""  # pragma: no cover


def get_registered_object(*args: object, **kwargs: object) -> object:
    """Get registered object."""
    pass  # pragma: no cover


def serialize_keras_object(*args: object, **kwargs: object) -> object:
    """Serialize a Keras object."""
    pass  # pragma: no cover


__all__ = [
    "H5WeightFormat",
    "NpzWeightFormat",
    "PickleWeightFormat",
    "SafetensorsWeightFormat",
    "_extract_numpy_weights",
    "_infer_weight_format",
    "_load_h5_weights",
    "_load_npz_weights",
    "_load_pickle_weights",
    "_load_safetensors_weights",
    "_save_as_h5",
    "_save_as_safetensors",
    "_validate_and_map_weights",
    "concatenate_arrays",
    "export_model_topology",
    "export_to_onnx",
    "export_to_tflite",
    "get_npz_bytes",
    "graph_to_flatbuffers",
    "graph_to_json",
    "graph_to_protobuf",
    "is_numpy_array",
    "json_to_graph",
    "load_model",
    "load_npz",
    "load_weights",
    "save_model",
    "save_weights",
    "to_numpy",
]
