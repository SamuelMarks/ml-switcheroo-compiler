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
            except Exception:
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
    import pickle

    try:
        from safetensors.numpy import save_file

        save_file(weights_np, filepath)
    except ImportError:
        with open(filepath, "wb") as f:
            pickle.dump(weights_np, f)


def _save_as_pickle(weights_np: dict, filepath: str) -> None:
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump(weights_np, f)


def save_weights(weights: dict, filepath: str) -> None:
    """Save weights to a file (h5, safetensors, or pickle)."""
    weights_np = _extract_numpy_weights(weights)

    if filepath.endswith(".h5"):
        _save_as_h5(weights_np, filepath)
    elif filepath.endswith(".safetensors"):
        _save_as_safetensors(weights_np, filepath)
    else:
        _save_as_pickle(weights_np, filepath)


def load_weights(filepath: str) -> dict:
    """Load weights from a file."""
    import pickle

    weights = {}
    if filepath.endswith(".h5"):
        try:
            import h5py

            with h5py.File(filepath, "r") as f:
                for k in f.keys():
                    weights[k] = f[k][()]
        except ImportError:
            with open(filepath, "rb") as f:
                weights = pickle.load(f)
    elif filepath.endswith(".safetensors"):
        try:
            from safetensors.numpy import load_file

            weights = load_file(filepath)
        except ImportError:
            with open(filepath, "rb") as f:
                weights = pickle.load(f)
    else:
        with open(filepath, "rb") as f:
            weights = pickle.load(f)
    return weights


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
        except Exception:
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
