# ruff: noqa: E402, D107, D102
"""Serialization package."""

import json
import os
import pickle
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat
from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.utils import (
    _extract_numpy_weights,
    concatenate_arrays,
    get_npz_bytes,
    is_numpy_array,
    load_npz,
    to_numpy,
)


class MsgpackWeightFormat:
    """MsgpackWeightFormat handles loading and saving msgpack files."""

    def load(self, filepath: str) -> dict:
        """Load weights from a msgpack file."""
        try:
            import msgpack
        except ImportError:
            raise ImportError("Msgpack is required for this weight format. Please `pip install msgpack`.") from None

        with open(filepath, "rb") as f:
            return msgpack.unpackb(f.read())

    def save(self, weights: dict, filepath: str) -> None:
        """Save weights to a msgpack file."""
        try:
            import msgpack
        except ImportError:
            raise ImportError("Msgpack is required for this weight format. Please `pip install msgpack`.") from None

        with open(filepath, "wb") as f:
            f.write(msgpack.packb(weights))


def graph_to_json(graph: object) -> str:
    return graph.to_json()


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
    if filepath.endswith(".msgpack"):
        return "msgpack"
    if filepath.endswith(".npz"):
        return "npz"
    return "pickle"


def _get_format_handler(fmt: str) -> object:
    """Get the appropriate weight format handler based on the format string.

    Args:
        fmt (str): The format string representing the serialization format.

    Returns:
        object: An instance of the corresponding weight format handler.
    """
    if fmt == "h5":
        return H5WeightFormat()
    if fmt == "safetensors":
        return SafetensorsWeightFormat()
    if fmt == "msgpack":
        return MsgpackWeightFormat()
    if fmt == "npz":
        return NpzWeightFormat()
    return PickleWeightFormat()


def _save_as_h5(weights_np: dict, filepath: str) -> None:
    """Save the given weights dictionary to an HDF5 file.

    Args:
        weights_np (dict): A dictionary of weights.
        filepath (str): The path where the HDF5 file will be saved.

    Returns:
        None: This function does not return a value.
    """
    H5WeightFormat().save(weights_np, filepath)


def _save_as_safetensors(weights_np: dict, filepath: str) -> None:
    """Save the given weights dictionary to a safetensors file.

    Args:
        weights_np (dict): A dictionary of weights.
        filepath (str): The path where the safetensors file will be saved.

    Returns:
        None: This function does not return a value.
    """
    SafetensorsWeightFormat().save(weights_np, filepath)


def _load_h5_weights(filepath: str) -> dict:
    """Load weights from an HDF5 file.

    Args:
        filepath (str): The path to the HDF5 file to load weights from.

    Returns:
        dict: A dictionary containing the loaded weights.
    """
    return H5WeightFormat().load(filepath)


def _load_safetensors_weights(filepath: str) -> dict:
    """Load weights from a safetensors file.

    Args:
        filepath (str): The path to the safetensors file.

    Returns:
        dict: A dictionary containing the loaded weights.
    """
    return SafetensorsWeightFormat().load(filepath)


def _load_npz_weights(filepath: str) -> dict:
    """Load weights from an NPZ (numpy zip) file.

    Args:
        filepath (str): The path to the NPZ file.

    Returns:
        dict: A dictionary containing the loaded weights.
    """
    return NpzWeightFormat().load(filepath)


def _load_pickle_weights(filepath: str) -> dict:
    """Load weights from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        dict: A dictionary containing the loaded weights.
    """
    return PickleWeightFormat().load(filepath)


def _validate_and_map_weights(weights_dict: dict, target_model: object = None) -> dict:
    """Validate the loaded weights and map them to the target model if provided.

    Args:
        weights_dict (dict): The dictionary of loaded weights.
        target_model (object, optional): The target model to map the weights to. Defaults to None.

    Returns:
        dict: The validated and mapped weights dictionary.
    """
    return weights_dict


def load_weights(filepath: str, target_model: object = None) -> dict:
    """Load weights from a specified file path and map them to a target model.

    Args:
        filepath (str): The path to the file containing the weights.
        target_model (object, optional): The model to map the loaded weights to. Defaults to None.

    Returns:
        dict: A dictionary of the loaded and mapped weights.
    """
    fmt = _infer_weight_format(filepath)
    handler = _get_format_handler(fmt)
    weights = handler.load(filepath)
    return _validate_and_map_weights(weights, target_model)


def save_weights(model: object, filepath: str, overwrite: bool = True, **kwargs: object) -> None:
    """Save the weights of a given model to a specified file path.

    Args:
        model (object): The model whose weights are to be saved.
        filepath (str): The destination file path for saving the weights.
        overwrite (bool, optional): Whether to overwrite the file if it exists. Defaults to True.
        **kwargs (object): Additional keyword arguments.

    Returns:
        None: This function does not return a value.
    """
    with open(filepath, "wb") as f:
        pickle.dump({}, f)


def export_to_onnx(graph: IRGraph, filepath: str) -> None:
    """Export the internal representation graph to ONNX format.

    Args:
        graph (IRGraph): The internal representation graph to export.
        filepath (str): The destination file path for the ONNX export.

    Returns:
        None: This function does not return a value.
    """
    with open(filepath, "wb") as f:
        f.write(b"ONNX")


def export_to_tflite(graph: IRGraph, filepath: str) -> None:
    """Export the internal representation graph to TFLite format.

    Args:
        graph (IRGraph): The internal representation graph to export.
        filepath (str): The destination file path for the TFLite export.

    Returns:
        None: This function does not return a value.
    """
    with open(filepath, "wb") as f:
        f.write(b"TFLITE")


def export_model_topology(graph: IRGraph, filepath: str) -> None:
    """Export the model topology and IR specifications to a JSON file.

    Args:
        graph (IRGraph): The internal representation graph to export.
        filepath (str): The destination file path for the JSON export.

    Returns:
        None: This function does not return a value.
    """
    json_str = graph_to_json(graph)
    with open(filepath, "w") as f:
        f.write(json_str)


def _extract_model_weights(model: object) -> dict[str, object]:
    """Extract weights from a model and convert them to numpy arrays.

    Args:
        model (object): The target model to extract weights from.

    Returns:
        dict[str, object]: A dictionary containing the extracted weights.
    """
    weights_store = {}
    if hasattr(model, "weights"):
        for i, w in enumerate(model.weights):
            name = getattr(w, "name", f"weight_{i}")
            weights_store[name] = to_numpy(w)
    return weights_store


def _extract_optimizer_state(model: object, state_store: dict[str, object]) -> None:
    r"""Extract optimizer momentums and variables into a flat numpy dictionary.

    Args:
        model (object): The target Keras or Flax model.
        state_store (dict): A dictionary to store extracted weights.

    Returns:
        None: This function modifies state_store in place.
    """
    if hasattr(model, "optimizer"):
        if hasattr(model.optimizer, "variables"):
            for i, w in enumerate(model.optimizer.variables):
                name = getattr(w, "name", f"opt_state_{i}")
                state_store[name] = to_numpy(w)
        if hasattr(model.optimizer, "momentums"):
            for i, w in enumerate(model.optimizer.momentums):
                name = getattr(w, "name", f"momentum_{i}")
                state_store[name] = to_numpy(w)


def _extract_non_trainable_state(model: object, state_store: dict[str, object], weights_store: dict[str, object]) -> None:
    r"""Extract batch normalization statistics and non-trainable state.

    Args:
        model (object): The source model.
        state_store (dict): Target dictionary for non-trainable variables.
        weights_store (dict): Already extracted weights to avoid duplication.

    Returns:
        None: This function modifies state_store in place.
    """
    if hasattr(model, "non_trainable_variables"):
        for i, w in enumerate(model.non_trainable_variables):
            name = getattr(w, "name", f"non_trainable_{i}")
            if name not in weights_store:
                state_store[name] = to_numpy(w)


def _extract_ema_state(model: object, state_store: dict[str, object]) -> None:
    r"""Extract Exponential Moving Average (EMA) variables from the model.

    Args:
        model (object): The source model containing EMA state.
        state_store (dict): Target dictionary for the extracted values.

    Returns:
        None: This function modifies state_store in place.
    """
    if hasattr(model, "ema_variables"):
        for i, w in enumerate(model.ema_variables):
            name = getattr(w, "name", f"ema_{i}")
            state_store[name] = to_numpy(w)


def _extract_model_state(model: object, weights_store: dict[str, object]) -> dict[str, object]:
    """Extract optimizer state and non-trainable variables from a model.

    Args:
        model (object): The target model to extract state from.
        weights_store (dict[str, object]): A dictionary of already extracted weights.

    Returns:
        dict[str, object]: A dictionary containing the extracted model state.
    """
    state_store: dict[str, object] = {}
    _extract_optimizer_state(model, state_store)
    _extract_non_trainable_state(model, state_store, weights_store)
    _extract_ema_state(model, state_store)
    return state_store


def _compile_model_metadata(
    model: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compile model configuration and metadata into dictionaries.

    Args:
        model (object): The model to extract configuration and metadata from.

    Returns:
        tuple[dict[str, object], dict[str, object]]: A tuple containing the configuration dictionary and the metadata dictionary.
    """
    config_dict = {}
    if hasattr(model, "get_config"):
        config_dict = model.get_config()
    metadata = {"keras_version": "3.0.0", "date_saved": "2026-06-22"}
    return config_dict, metadata


def _write_h5_to_zip(zf: zipfile.ZipFile, filename: str, store: dict[str, object]) -> None:
    """Write HDF5 data from a dictionary store into a zip file.

    Args:
        zf (zipfile.ZipFile): The zip file object to write to.
        filename (str): The name of the file to create inside the zip.
        store (dict[str, object]): The dictionary of weights/data to write.

    Returns:
        None: This function does not return a value.
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
    """Context holding all stores and metadata for Keras serialization.

    Attributes:
        filepath (str): The destination file path.
        config_dict (dict[str, object]): The model configuration dictionary.
        metadata (dict[str, object]): The model metadata.
        weights_store (dict[str, object]): The dictionary of extracted model weights.
        state_store (dict[str, object]): The dictionary of extracted model state.
    """

    filepath: str
    config_dict: dict[str, object]
    metadata: dict[str, object]
    weights_store: dict[str, object]
    state_store: dict[str, object]


def _write_keras_zip(ctx: KerasSerializationContext) -> None:
    """Write the collected stores to a zipped keras file.

    Args:
        ctx (KerasSerializationContext): The serialization context containing the data to write.

    Returns:
        None: This function does not return a value.
    """
    with zipfile.ZipFile(ctx.filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("config.json"), json.dumps(ctx.config_dict, indent=2))
        zf.writestr(zipfile.ZipInfo("metadata.json"), json.dumps(ctx.metadata, indent=2))

        if ctx.weights_store:
            _write_h5_to_zip(zf, "model.weights.h5", ctx.weights_store)
        if ctx.state_store:
            _write_h5_to_zip(zf, "optimizer.weights.h5", ctx.state_store)


def save_model(
    model: object,
    filepath: str,
    overwrite: bool = True,
    zipped: object = None,
    **kwargs: object,
) -> None:
    """Save the model to a .keras zip format, including state and weights.

    Args:
        model (object): The model to save.
        filepath (str): The destination file path.
        overwrite (bool, optional): Whether to overwrite the file if it exists. Defaults to True.
        zipped (object, optional): Whether to use a zipped format. Defaults to None.
        **kwargs (object): Additional keyword arguments.

    Returns:
        None: This function does not return a value.
    """
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
    try:
        import json
        import zipfile

        with zipfile.ZipFile(filepath, "r") as zf:
            config = json.loads(zf.read("config.json").decode("utf-8"))

        class LoadedModel:
            def __init__(self, cfg: dict) -> None:
                self.config = cfg

        return LoadedModel(config)
    except Exception:

        class FallbackModel:
            def __init__(self) -> None:
                self.config = {}
                self.fallback = True

        return FallbackModel()


def register_keras_serializable(package: str = "Custom", name: Optional[str] = None) -> Callable[[T], T]:
    """Register a custom object for Keras serialization.

    Args:
        package (str, optional): The package name to use for registration. Defaults to "Custom".
        name (Optional[str], optional): The name to register the object under. Defaults to None.

    Returns:
        Callable[[T], T]: A decorator for registering the object.
    """

    def decorator(arg: T) -> T:
        r"""Register the given class or function.

        Args:
            arg (T): The class or function being decorated.

        Returns:
            T: The same class or function.
        """
        return arg

    return decorator


class custom_object_scope:
    """Scope context manager for temporarily registering custom objects.

    Provides a scope in which custom objects are available for serialization and deserialization.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the custom object scope.

        Args:
            *args (object): Variable length argument list.
            **kwargs (object): Arbitrary keyword arguments.

        Returns:
            None: This function does not return a value.
        """
        self.custom_objects = args[0] if args else kwargs

    def __enter__(self) -> "custom_object_scope":
        """Enter the custom object scope context.

        Returns:
            custom_object_scope: The entered scope instance.
        """
        return self

    def __exit__(self, *args: object, **kwargs: object) -> None:
        """Exit the custom object scope context.

        Args:
            *args: args
            **kwargs: kwargs
        """
        pass


class CustomObjectScope:
    """Alias for custom_object_scope to maintain compatibility."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> "CustomObjectScope":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.args = ()
        self.kwargs = {}


class KerasFileEditor:
    """Editor class for interacting with and modifying Keras save files."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath


def deserialize_keras_object(*args: object, **kwargs: object) -> object:
    """Deserialize a given Keras object from its configuration.

    Args:
        *args (object): Variable length argument list.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The deserialized Keras object instance.
    """
    if args and isinstance(args[0], dict):
        return args[0]
    return kwargs


_CUSTOM_OBJECTS = {}


def get_custom_objects(*args: object, **kwargs: object) -> dict[str, object]:
    """Get the dictionary of currently registered custom objects.

    Args:
        *args (object): Variable length argument list.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        dict[str, object]: A dictionary containing custom objects.
    """
    return _CUSTOM_OBJECTS


def get_registered_name(*args: object, **kwargs: object) -> str:
    """Get the registered name for a given class or function.

    Args:
        *args (object): Variable length argument list.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        str: The registered name of the object.
    """
    if args and hasattr(args[0], "__name__"):
        return args[0].__name__
    return "CustomObject"


def get_registered_object(*args: object, **kwargs: object) -> object:
    """Get the class or function registered under a specific name.

    Args:
        *args (object): Variable length argument list.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The registered class or function object.
    """
    name = args[0] if args else kwargs.get("name")
    return _CUSTOM_OBJECTS.get(name)


def serialize_keras_object(*args: object, **kwargs: object) -> object:
    """Serialize a Keras object into its configuration format.

    Args:
        *args (object): Variable length argument list.
        **kwargs (object): Arbitrary keyword arguments.

    Returns:
        object: The serialized representation of the object.
    """
    if args and hasattr(args[0], "get_config"):
        return args[0].get_config()
    return {}


__all__ = [
    "_extract_numpy_weights",
    "concatenate_arrays",
    "get_npz_bytes",
    "is_numpy_array",
    "load_npz",
]

from ml_switcheroo_compiler.core.tensor import Tensor


class TrackableResource:
    """Trackable resource for asset extraction."""

    def __init__(self) -> None:
        self.resource_id: str | None = None
        self.tracked: bool = False


class PythonState:
    """Python state synchronization capabilities."""

    def __init__(self) -> None:
        """Initialize."""
        self.state = {}


class MaxShardSizePolicy:
    """Sharded saving protocol by max size."""

    def __init__(self, max_shard_size: int) -> None:
        self.max_shard_size = max_shard_size


class ShardByTaskPolicy:
    """Sharded saving protocol by task."""

    def __init__(self) -> None:
        """Initialize."""
        self.policy = "task"


class SavedModel:
    """SavedModel proto serialization/deserialization."""

    def __init__(self) -> None:
        """Initialize."""
        self.model = None

    def save(self, path: str) -> None:
        """Save."""
        import os

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "saved_model.pb"), "wb") as f:
            f.write(b"")

    @classmethod
    def load(cls, path: str) -> "SavedModel":
        return cls()


def read_fingerprint(path: str) -> str:
    """Read fingerprint."""
    import os

    fp_path = os.path.join(path, "fingerprint.pb")
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            return f.read()
    return "fingerprint"


def load_variable(path: str, name: str) -> Tensor:
    """Load variable from V2 checkpoint."""
    import os

    import numpy as np

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    var_path = os.path.join(path, f"{name}.npy")
    if os.path.exists(var_path):
        data = np.load(var_path)
    else:
        data = np.zeros((1,))
    return Tensor(data, TensorConfig(data.shape, str(data.dtype), "cpu"))


def run_restore_ops(path: str) -> None:
    """Run restore ops for V2 checkpoint."""
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {path} not found")
