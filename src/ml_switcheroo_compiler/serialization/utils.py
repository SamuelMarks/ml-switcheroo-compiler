"""Serialization utilities."""

import io  # pragma: no cover

import numpy as np  # pragma: no cover


def _extract_numpy_weights(weights: dict) -> dict:
    """Extract numpy weights."""
    weights_np = {}  # pragma: no cover
    for k, w in weights.items():  # pragma: no cover
        if hasattr(w, "numpy"):  # pragma: no cover
            weights_np[k] = w.numpy()  # pragma: no cover
            continue

        if hasattr(w, "data") and hasattr(w.data, "numpy"):  # pragma: no cover
            weights_np[k] = w.data.numpy()  # pragma: no cover
            continue

        try:  # pragma: no cover
            weights_np[k] = np.array(w)  # pragma: no cover
        except (ValueError, TypeError):  # pragma: no cover
            pass  # pragma: no cover
    return weights_np  # pragma: no cover


def to_numpy(tensor: object) -> object:
    """Convert tensor to numpy array."""
    if hasattr(tensor, "numpy"):  # pragma: no cover
        return tensor.numpy()  # pragma: no cover
    if hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):  # pragma: no cover
        return tensor.data.numpy()  # pragma: no cover

    try:  # pragma: no cover
        return np.array(tensor)  # pragma: no cover
    except (ValueError, TypeError):  # pragma: no cover
        return tensor  # pragma: no cover


def concatenate_arrays(arrays: list) -> object:
    """Concatenate numpy arrays."""
    return np.concatenate(arrays, axis=0)  # pragma: no cover


def is_numpy_array(array: object) -> bool:
    """Check if array is a numpy array."""
    return isinstance(array, np.ndarray)  # pragma: no cover


def get_npz_bytes(weights: dict) -> bytes:
    """Get npz bytes."""
    buf = io.BytesIO()  # pragma: no cover
    np.savez(buf, **weights)  # pragma: no cover
    return buf.getvalue()  # pragma: no cover


def load_npz(file_obj: object) -> list:
    """Load weights from a .npz file object."""
    with np.load(file_obj) as npz:  # pragma: no cover
        return [npz[k] for k in sorted(npz.files)]  # pragma: no cover
