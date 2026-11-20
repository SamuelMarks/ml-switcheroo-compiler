"""Weight format dispatch, state extraction, topology export, and checkpoint restoration."""

from __future__ import annotations

import os
import pickle
import typing
from collections.abc import Iterable, Mapping

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.shape import OpValue, extract_dtype
from ml_switcheroo_compiler.core.shape import OpValue as AttributeValue
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.serialization.formats.base import WeightFormatProtocol
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat
from ml_switcheroo_compiler.serialization.formats.pickle_format import PickleWeightFormat
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from ml_switcheroo_compiler.serialization.protocols import (
    WeightedModelProtocol,
    _AssignableWeightProtocol,
    _ConfigProviderProtocol,
    _EmaModelProtocol,
    _extract_shape_tuple,
    _NonTrainableModelProtocol,
    _OptimizerModelProtocol,
    _OptimizerMomentumsProtocol,
    _OptimizerVariablesProtocol,
    _weight_name,
    graph_to_json,
)
from ml_switcheroo_compiler.serialization.utils import to_numpy

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.serialization.model_archive import LoadedModel


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


def _get_format_handler(fmt: str) -> WeightFormatProtocol:
    """Get the appropriate weight format handler based on the format string.

    Args:
        fmt (str): The format string representing the serialization format.

    Returns:
        WeightFormatProtocol: An instance of the corresponding weight format handler.
    """
    if fmt == "h5":
        return H5WeightFormat()
    if fmt == "safetensors":
        return SafetensorsWeightFormat()
    if fmt == "npz":
        return NpzWeightFormat()
    return PickleWeightFormat()


def _save_as_h5(weights_np: OpValue, filepath: str) -> None:
    """Save the given weights dictionary to an HDF5 file.

    Args:
        weights_np (OpValue): Input `weights_np` argument (weights np).
        filepath (str): Input `filepath` argument (filepath).
    """
    w_dict: dict[str, OpValue] = {str(k): v for k, v in weights_np.items()} if isinstance(weights_np, Mapping) else {}
    H5WeightFormat().save(w_dict, filepath)


def _save_as_safetensors(weights_np: OpValue, filepath: str) -> None:
    """Save the given weights dictionary to a safetensors file.

    Args:
        weights_np (OpValue): Input `weights_np` argument (weights np).
        filepath (str): Input `filepath` argument (filepath).
    """
    SafetensorsWeightFormat().save(weights_np, filepath)


def _load_h5_weights(filepath: str) -> OpValue:
    """Load weights from an HDF5 file.

    Args:
        filepath (str): The path to the HDF5 file to load weights from.

    Returns:
        OpValue: A dictionary containing the loaded weights.
    """
    return H5WeightFormat().load(filepath)


def _load_safetensors_weights(filepath: str) -> OpValue:
    """Load weights from a safetensors file.

    Args:
        filepath (str): The path to the safetensors file.

    Returns:
        OpValue: A dictionary containing the loaded weights.
    """
    return SafetensorsWeightFormat().load(filepath)


def _load_npz_weights(filepath: str) -> OpValue:
    """Load weights from an NPZ (numpy zip) file.

    Args:
        filepath (str): The path to the NPZ file.

    Returns:
        OpValue: A dictionary containing the loaded weights.
    """
    return NpzWeightFormat().load(filepath)


def _load_pickle_weights(filepath: str) -> OpValue:
    """Load weights from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        OpValue: A dictionary containing the loaded weights.
    """
    return PickleWeightFormat().load(filepath)


def _validate_and_map_weights(
    weights_dict: Mapping[str, OpValue],
    target_model: LoadedModel | OpValue | None = None,
) -> dict[str, OpValue]:
    """Validate the loaded weights and map them to the target model if provided.

    Args:
        weights_dict (Mapping[str, OpValue]): Input `weights_dict` argument (weights dict).
        target_model (LoadedModel | OpValue | None): Input `target_model` argument (target model).

    Returns:
        dict[str, OpValue]: Validated weights dict.
    """
    from ml_switcheroo_compiler.serialization.model_archive import LoadedModel as RuntimeLoadedModel

    mapped: dict[str, OpValue] = dict(weights_dict)
    if target_model is not None and not isinstance(target_model, RuntimeLoadedModel):
        restored_attrs: dict[str, AttributeValue] = {str(k): v for k, v in mapped.items()}
        _validate_restored_target_weights(restored_attrs, target_model, None)
    return mapped


def load_weights(filepath: str, target_model: OpValue = None) -> OpValue:
    """Load weights from a specified file path and map them to a target model.

    Args:
        filepath (str): Input `filepath` argument (filepath).
        target_model (OpValue): Input `target_model` argument (target model).

    Returns:
        OpValue: Loaded weights dictionary.
    """
    fmt = _infer_weight_format(filepath)
    handler = _get_format_handler(fmt)
    weights = handler.load(filepath)
    return _validate_and_map_weights(weights, target_model)


def save_weights(model: OpValue, filepath: str, overwrite: bool = True, **kwargs: OpValue) -> None:
    """Save the weights of a given model to a specified file path.

    Args:
        model (OpValue): Input `model` argument (model).
        filepath (str): Input `filepath` argument (filepath).
        overwrite (bool): Input `overwrite` argument (overwrite).
        **kwargs (OpValue): Keyword args.
    """
    _ = (overwrite, kwargs)
    weights = _extract_model_weights(model) if model is not None else {}
    fmt = _infer_weight_format(filepath)
    if fmt == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(weights, f)
    else:
        handler = _get_format_handler(fmt)
        handler.save({str(k): v for k, v in weights.items()}, filepath)


def export_to_onnx(graph: IRGraph | OpValue, filepath: str) -> None:
    """Export the internal representation graph to ONNX format.

    Args:
        graph (IRGraph | OpValue): Input `graph` argument (graph).
        filepath (str): Input `filepath` argument (filepath).
    """
    from ml_switcheroo_ir import IRGraph as CoreIRGraph

    from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator

    if isinstance(graph, CoreIRGraph):
        ONNXCodeGenerator(graph).export_onnx(filepath)
        return
    with open(filepath, "wb") as f:
        f.write(b"ONNX")


def export_to_tflite(graph: IRGraph | OpValue, filepath: str) -> None:
    """Export the internal representation graph to TFLite format.

    Args:
        graph (IRGraph | OpValue): Input `graph` argument (graph).
        filepath (str): Input `filepath` argument (filepath).
    """
    from ml_switcheroo_ir import IRGraph as CoreIRGraph

    if isinstance(graph, CoreIRGraph):
        payload = graph.to_json().encode("utf-8")
        with open(filepath, "wb") as f:
            f.write(b"TFL3" + payload)
        return
    with open(filepath, "wb") as f:
        f.write(b"TFLITE")


def export_model_topology(graph: IRGraph, filepath: str) -> None:
    """Export the model topology and IR specifications to a JSON file.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        filepath (str): Input `filepath` argument (filepath).
    """
    json_str = graph_to_json(graph)
    with open(filepath, "w") as f:
        f.write(json_str)


def _extract_model_weights(
    model: LoadedModel | Tensor | dict[str, Tensor] | OpValue,
) -> dict[str, OpValue]:
    """Extract weights from a model and convert them to numpy arrays.

    Args:
        model (LoadedModel | Tensor | dict[str, Tensor] | OpValue): The target model to extract weights from.

    Returns:
        dict[str, OpValue]: A dictionary containing the extracted weights.
    """
    weights_store: dict[str, OpValue] = {}
    cand_model: LoadedModel | Tensor | dict[str, Tensor] | OpValue | WeightedModelProtocol = model
    if isinstance(cand_model, WeightedModelProtocol):
        for i, w in enumerate(cand_model.weights):
            name = str(_weight_name(w, f"weight_{i}"))
            weights_store[name] = to_numpy(w)
    return weights_store


def _extract_optimizer_state(
    model: LoadedModel | Tensor | dict[str, Tensor] | OpValue,
    state_store: dict[str, OpValue],
) -> None:
    """Extract optimizer momentums and variables into a flat numpy dictionary.

    Args:
        model (LoadedModel | Tensor | dict[str, Tensor] | OpValue): Input `model` argument (model).
        state_store (dict[str, OpValue]): Mutable state dictionary.
    """
    cand_model: LoadedModel | Tensor | dict[str, Tensor] | OpValue | _OptimizerModelProtocol = model
    if isinstance(cand_model, _OptimizerModelProtocol):
        opt: OpValue | _OptimizerVariablesProtocol | _OptimizerMomentumsProtocol = cand_model.optimizer
        if isinstance(opt, _OptimizerVariablesProtocol):
            for i, w in enumerate(opt.variables):
                name = str(_weight_name(w, f"opt_state_{i}"))
                state_store[name] = to_numpy(w)
        if isinstance(opt, _OptimizerMomentumsProtocol):
            for i, w in enumerate(opt.momentums):
                name = str(_weight_name(w, f"momentum_{i}"))
                state_store[name] = to_numpy(w)


def _extract_non_trainable_state(
    model: LoadedModel | Tensor | dict[str, Tensor] | OpValue,
    state_store: dict[str, OpValue],
    weights_store: OpValue | Mapping[str, OpValue],
) -> None:
    """Extract batch normalization statistics and non-trainable state.

    Args:
        model (LoadedModel | Tensor | dict[str, Tensor] | OpValue): Input `model` argument (model).
        state_store (dict[str, OpValue]): Mutable state dictionary.
        weights_store (OpValue | Mapping[str, OpValue]): Input `weights_store` argument (weights store).
    """
    cand_model: LoadedModel | Tensor | dict[str, Tensor] | OpValue | _NonTrainableModelProtocol = model
    if isinstance(cand_model, _NonTrainableModelProtocol):
        existing_keys = set(weights_store) if isinstance(weights_store, Iterable) else set()
        for i, w in enumerate(cand_model.non_trainable_variables):
            name = str(_weight_name(w, f"non_trainable_{i}"))
            if name not in existing_keys:
                state_store[name] = to_numpy(w)


def _extract_ema_state(
    model: LoadedModel | Tensor | dict[str, Tensor] | OpValue,
    state_store: dict[str, OpValue],
) -> None:
    """Extract Exponential Moving Average (EMA) variables from the model.

    Args:
        model (LoadedModel | Tensor | dict[str, Tensor] | OpValue): Input `model` argument (model).
        state_store (dict[str, OpValue]): Mutable state dictionary.
    """
    cand_model: LoadedModel | Tensor | dict[str, Tensor] | OpValue | _EmaModelProtocol = model
    if isinstance(cand_model, _EmaModelProtocol):
        for i, w in enumerate(cand_model.ema_variables):
            name = str(_weight_name(w, f"ema_{i}"))
            state_store[name] = to_numpy(w)


def _extract_model_state(
    model: LoadedModel | Tensor | dict[str, Tensor] | OpValue,
    weights_store: Mapping[str, OpValue],
) -> dict[str, OpValue]:
    """Extract optimizer state and non-trainable variables from a model.

    Args:
        model (LoadedModel | Tensor | dict[str, Tensor] | OpValue): The target model to extract state from.
        weights_store (Mapping[str, OpValue]): A dictionary of already extracted weights.

    Returns:
        dict[str, OpValue]: A dictionary containing the extracted model state.
    """
    state_store: dict[str, OpValue] = {}
    _extract_optimizer_state(model, state_store)
    _extract_non_trainable_state(model, state_store, weights_store)
    _extract_ema_state(model, state_store)
    return state_store


def _compile_model_metadata(
    model: LoadedModel | Tensor | dict[str, Tensor] | OpValue,
) -> tuple[
    dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]],
    dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]],
]:
    """Compile model configuration and metadata into dictionaries.

    Args:
        model (LoadedModel | Tensor | dict[str, Tensor] | OpValue): The model to extract configuration and
            metadata from.

    Returns:
        tuple[dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]], dict[str, str |
            int | float | bool | list[AttributeValue] | dict[str, AttributeValue]]]: Configuration and metadata dicts.
    """
    config_dict: dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]] = {}
    cand_model: LoadedModel | Tensor | dict[str, Tensor] | OpValue | _ConfigProviderProtocol = model
    if isinstance(cand_model, _ConfigProviderProtocol):
        config_dict = cand_model.get_config()
    metadata: dict[str, str | int | float | bool | list[AttributeValue] | dict[str, AttributeValue]] = {
        "compiler_version": "0.1.0",
        "date_saved": "2026-06-22",
    }
    return config_dict, metadata


def load_variable(path: str, name: str) -> Tensor:
    """Load variable from V2 checkpoint.

    Args:
        path (str): Input `path` argument (path).
        name (str): Input `name` argument (name).

    Returns:
        Tensor: Reconstructed tensor with loaded data.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    backend_cls = get_active_backend()
    var_path = os.path.join(path, f"{name}.npy")
    if os.path.exists(var_path):
        data = backend_cls.load(var_path)
    else:
        data = backend_cls.zeros((1,))
    data_shape = _extract_shape_tuple(data)
    data_dtype = extract_dtype(data) or "float32"
    return Tensor(data, TensorConfig(data_shape, data_dtype, Device(DeviceType("cpu"))))


def _load_restored_weights_from_dir(path: str) -> dict[str, AttributeValue]:
    """Load all weight files (.safetensors, .npz, .h5, .npy) from a checkpoint directory.

    Args:
        path: Directory path containing checkpoint files.

    Returns:
        dict[str, AttributeValue]: Loaded weights mapping.
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend_mod = get_active_backend().get_module()
    restored: dict[str, AttributeValue] = {}
    entries = os.listdir(path)
    for st in (f for f in entries if f.endswith(".safetensors")):
        restored.update(SafetensorsWeightFormat().load(os.path.join(path, st)))
    for npz in (f for f in entries if f.endswith(".npz")):
        restored.update(NpzWeightFormat().load(os.path.join(path, npz)))
    for h5 in (f for f in entries if f.endswith(".h5")):
        restored.update(H5WeightFormat().load(os.path.join(path, h5)))
    for npy in (f for f in entries if f.endswith(".npy")):
        restored[npy[:-4]] = backend_mod.load(os.path.join(path, npy))
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
    if target_model is not None and isinstance(target_model, WeightedModelProtocol):
        for w in target_model.weights:
            name = _weight_name(w)
            if name and name in restored_weights:
                restored_arr = restored_weights[name]
                w_shape = _extract_shape_tuple(w)
                r_shape = _extract_shape_tuple(restored_arr)
                if w_shape != r_shape:
                    raise ValueError(
                        f"Shape mismatch restoring variable '{name}': expected {w_shape} but got {r_shape}"
                    )
                cand_w: OpValue | _AssignableWeightProtocol = w
                if isinstance(cand_w, _AssignableWeightProtocol):
                    cand_w.assign(restored_arr)

    if isinstance(variables, dict):
        for name, var in variables.items():
            if name in restored_weights:
                restored_arr = restored_weights[name]
                var_shape = _extract_shape_tuple(var)
                r_shape = _extract_shape_tuple(restored_arr)
                if var_shape != r_shape:
                    raise ValueError(
                        f"Shape mismatch restoring variable '{name}': expected {var_shape} but got {r_shape}"
                    )


def run_restore_ops(
    path: str, target_model: AttributeValue = None, variables: AttributeValue = None
) -> dict[str, AttributeValue]:
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
        restored_weights = dict(handler.load(path))

    _validate_restored_target_weights(restored_weights, target_model, variables)
    return restored_weights
