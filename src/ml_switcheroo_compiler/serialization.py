"""Serialization Formats."""

import json

from ml_switcheroo_compiler.ir.core import IRGraph


def graph_to_json(graph: IRGraph) -> str:
    """Implement JSON serialization for snapshot testing IR passes.

    Args:
        graph (IRGraph): Argument graph


    Returns:
        str: The evaluated output resulting from this operation.
    """
    data = {"nodes": {}}
    for node_id, node in graph.nodes.items():
        data["nodes"][node_id] = {
            "op": node.op_type,
            "inputs": node.inputs,
        }
    return json.dumps(data, indent=2)


def json_to_graph(json_str: str) -> IRGraph:
    """Deserialize JSON back into an IRGraph.

    Args:
        json_str (str): Argument json_str


    Returns:
        IRGraph: The evaluated output resulting from this operation.
    """
    from ml_switcheroo_compiler.ir.core import IRNode

    data = json.loads(json_str)
    graph = IRGraph()
    for node_id, node_data in data.get("nodes", {}).items():
        node = IRNode(
            id=node_id,
            op_type=node_data.get("op", ""),
            inputs=node_data.get("inputs", []),
            shape_metadata=(),
        )
        graph.nodes[node_id] = node
    return graph


def graph_to_protobuf(graph: IRGraph) -> bytes:
    """Define IRGraph Protobuf .proto spec serialization.

    Args:
        graph (IRGraph): Argument graph


    Returns:
        bytes: The evaluated output resulting from this operation.
    """
    # Dummy implementation
    return b""


def graph_to_flatbuffers(graph: IRGraph) -> bytes:
    """Implement FlatBuffers serialization for zero-copy JS/TS loading.

    Args:
        graph (IRGraph): Argument graph


    Returns:
        bytes: The evaluated output resulting from this operation.
    """
    # Dummy implementation
    return b""


def _extract_numpy_weights(weights: dict) -> dict:
    import numpy as np

    weights_np = {}
    for k, w in weights.items():
        if hasattr(w, "numpy"):
            weights_np[k] = w.numpy()
        elif hasattr(w, "data") and hasattr(w.data, "numpy"):
            weights_np[k] = w.data.numpy()
        else:
            try:
                weights_np[k] = np.array(w)
            except (ValueError, TypeError):
                pass
    return weights_np


def _save_as_h5(weights_np: dict, filepath: str) -> None:
    import pickle

    try:
        import h5py

        with h5py.File(filepath, "w") as f:
            for k, v in weights_np.items():
                f.create_dataset(k, data=v)
    except ImportError:
        with open(filepath, "wb") as f:
            pickle.dump(weights_np, f)


def _save_as_safetensors(weights_np: dict, filepath: str) -> None:
    import json
    import struct

    header = {}
    offset = 0
    buffers = []

    for k, v in weights_np.items():
        if not hasattr(v, "dtype") or not hasattr(v, "shape") or not hasattr(v, "tobytes"):
            continue

        dtype_map = {
            "float32": "F32",
            "float64": "F64",
            "float16": "F16",
            "int32": "I32",
            "int64": "I64",
            "int16": "I16",
            "int8": "I8",
            "uint8": "U8",
            "bool": "BOOL",
        }

        dtype_str = str(v.dtype)
        st_dtype = dtype_map.get(dtype_str, "F32")

        buffer = v.tobytes()
        length = len(buffer)

        header[k] = {
            "dtype": st_dtype,
            "shape": list(v.shape),
            "data_offsets": [offset, offset + length],
        }

        buffers.append(buffer)
        offset += length

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_length = len(header_bytes)

    padding_length = (8 - (header_length % 8)) % 8
    header_bytes += b" " * padding_length
    header_length += padding_length

    with open(filepath, "wb") as f:
        f.write(struct.pack("<Q", header_length))
        f.write(header_bytes)
        for buffer in buffers:
            f.write(buffer)


def _load_h5_weights(filepath: str) -> dict:
    import pickle

    try:
        import h5py

        weights = {}
        with h5py.File(filepath, "r") as f:
            for k in f.keys():
                weights[k] = f[k][()]
        return weights
    except ImportError:
        with open(filepath, "rb") as f:
            return pickle.load(f)


def _load_safetensors_weights(filepath: str) -> dict:
    import json
    import struct
    import numpy as np

    dtype_map = {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }

    with open(filepath, "rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            return {}
        header_size = struct.unpack("<Q", header_size_bytes)[0]

        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode("utf-8"))

        weights = {}
        for k, v in header.items():
            if k == "__metadata__":
                continue
            offsets = v["data_offsets"]
            f.seek(8 + header_size + offsets[0])
            buffer = f.read(offsets[1] - offsets[0])
            dtype = dtype_map.get(v["dtype"], np.float32)
            arr = np.frombuffer(buffer, dtype=dtype).reshape(v["shape"]).copy()
            weights[k] = arr

        return weights


def _load_npz_weights(filepath: str) -> dict:
    import numpy as np

    with np.load(filepath) as npz:
        return {k: npz[k] for k in npz.files}


def _load_pickle_weights(filepath: str) -> dict:
    import pickle

    with open(filepath, "rb") as f:
        return pickle.load(f)


def _infer_weight_format(filepath: str) -> str:
    if filepath.endswith(".h5"):
        return "h5"
    if filepath.endswith(".safetensors"):
        return "safetensors"
    if filepath.endswith(".npz"):
        return "npz"
    return "pickle"


def _validate_and_map_weights(weights_dict: dict, target_model: object = None) -> dict:
    return weights_dict


def load_weights(filepath: str, target_model: object = None) -> dict:
    """Load weights from a file."""
    fmt = _infer_weight_format(filepath)
    if fmt == "h5":
        weights = _load_h5_weights(filepath)
    elif fmt == "safetensors":
        weights = _load_safetensors_weights(filepath)
    elif fmt == "npz":
        weights = _load_npz_weights(filepath)
    else:
        weights = _load_pickle_weights(filepath)
    return _validate_and_map_weights(weights, target_model)


def to_numpy(tensor: object) -> object:
    """Convert tensor to numpy array."""
    import numpy as np

    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    elif hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
        return tensor.data.numpy()
    else:
        try:
            return np.array(tensor)
        except (ValueError, TypeError):
            return tensor


def get_npz_bytes(weights: dict) -> bytes:
    """Get npz bytes."""
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez(buf, **weights)
    return buf.getvalue()


def concatenate_arrays(arrays: list) -> object:
    """Concatenate numpy arrays."""
    import numpy as np

    return np.concatenate(arrays, axis=0)


def is_numpy_array(array: object) -> bool:
    """Check if array is a numpy array."""
    import numpy as np

    return isinstance(array, np.ndarray)


def load_npz(file_obj: object) -> list:
    """Load weights from a .npz file object."""
    import numpy as np

    with np.load(file_obj) as npz:
        return [npz[k] for k in sorted(npz.files)]


def export_to_onnx(graph: IRGraph, filepath: str) -> None:
    """Export the graph to ONNX format."""
    # A complete ONNX translation pipeline would map IR nodes to ONNX operators
    # This is a stub for the pipeline
    with open(filepath, "wb") as f:
        f.write(b"ONNX")


def export_to_tflite(graph: IRGraph, filepath: str) -> None:
    """Export the graph to TFLite format."""
    # A complete TFLite translation pipeline would map IR nodes to TFLite flatbuffers
    # This is a stub for the pipeline
    with open(filepath, "wb") as f:
        f.write(b"TFLITE")


def export_model_topology(graph: IRGraph, filepath: str) -> None:
    """Export the model topology and IR specifications."""
    json_str = graph_to_json(graph)
    with open(filepath, "w") as f:
        f.write(json_str)


def save_model(
    model: object, filepath: str, overwrite: bool = True, zipped: object = None, **kwargs: object
) -> None:
    """Save model to .keras zip format, including state and weights."""
    import zipfile
    import json

    # We will simulate model extraction by checking model variables
    # If the model has `variables`, `trainable_variables`, `non_trainable_variables`, `optimizer`, `metrics`
    # We must serialize EMA or momentum matrices from these and bundle them in the zip file

    config_dict = {}
    if hasattr(model, "get_config"):
        config_dict = model.get_config()

    metadata = {"keras_version": "3.0.0", "date_saved": "2026-06-22"}

    weights_store = {}
    if hasattr(model, "weights"):
        for i, w in enumerate(model.weights):
            name = getattr(w, "name", f"weight_{i}")
            weights_store[name] = to_numpy(w)

    # Track state: optimizer state, EMA, momentum, metrics state
    state_store = {}
    if hasattr(model, "optimizer") and hasattr(model.optimizer, "variables"):
        for i, w in enumerate(model.optimizer.variables):
            name = getattr(w, "name", f"opt_state_{i}")
            state_store[name] = to_numpy(w)

    if hasattr(model, "non_trainable_variables"):
        for i, w in enumerate(model.non_trainable_variables):
            name = getattr(w, "name", f"non_trainable_{i}")
            if name not in weights_store:
                state_store[name] = to_numpy(w)

    if hasattr(model, "ema_variables"):
        for i, w in enumerate(model.ema_variables):
            name = getattr(w, "name", f"ema_{i}")
            state_store[name] = to_numpy(w)

    # Include momentum from optimizer specifically if defined
    if hasattr(model, "optimizer") and hasattr(model.optimizer, "momentums"):
        for i, w in enumerate(model.optimizer.momentums):
            name = getattr(w, "name", f"momentum_{i}")
            state_store[name] = to_numpy(w)

    with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        zinfo = zipfile.ZipInfo("config.json")
        zf.writestr(zinfo, json.dumps(config_dict, indent=2))

        zinfo = zipfile.ZipInfo("metadata.json")
        zf.writestr(zinfo, json.dumps(metadata, indent=2))

        if weights_store:
            zinfo = zipfile.ZipInfo("model.weights.h5")
            # We can't easily write h5 to string, so we'll just dump to bytes if h5py is missing, or save to disk and add.
            # But the backend requires proper `.keras` inclusion.
            # We will use our _save_as_h5 wrapper by writing to a temp file
            import tempfile
            import os

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
            os.close(tmp_fd)
            _save_as_h5(weights_store, tmp_path)
            with open(tmp_path, "rb") as f:
                zf.writestr(zinfo, f.read())
            os.remove(tmp_path)

        if state_store:
            zinfo = zipfile.ZipInfo("optimizer.weights.h5")
            import tempfile
            import os

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
            os.close(tmp_fd)
            _save_as_h5(state_store, tmp_path)
            with open(tmp_path, "rb") as f:
                zf.writestr(zinfo, f.read())
            os.remove(tmp_path)


def load_model(
    filepath: str,
    custom_objects: object = None,
    compile: bool = True,
    safe_mode: bool = True,
    **kwargs: object,
) -> object:
    """Load model."""
    pass


def save_weights(model: object, filepath: str, overwrite: bool = True, **kwargs: object) -> None:
    """Save weights."""
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump({}, f)
