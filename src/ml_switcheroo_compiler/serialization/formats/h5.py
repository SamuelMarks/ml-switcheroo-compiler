"""H5 format serialization."""

import h5py

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class H5WeightFormat(WeightLoader, WeightSaver):
    """H5 weight format handler."""

    def load(self, filepath: str) -> dict:
        """Load h5 weights."""
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        if hasattr(backend, "load_h5"):
            return backend.load_h5(filepath)

        result = {}
        with h5py.File(filepath, "r") as f:

            def _visit(name: str, node: object) -> None:
                if isinstance(node, h5py.Dataset):
                    result[name] = node[()]  # h5py returns numpy arrays from slicing

            f.visititems(_visit)
        return result

    def save(self, weights_np: dict, filepath: str) -> None:
        """Save h5 weights."""
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        if hasattr(backend, "save_h5"):
            backend.save_h5(weights_np, filepath)
            return

        with h5py.File(filepath, "w") as f:
            for k, v in weights_np.items():
                if hasattr(v, "numpy"):
                    v = v.numpy()
                elif hasattr(v, "data") and hasattr(v.data, "numpy"):
                    v = v.data.numpy()
                elif hasattr(v, "tolist"):
                    v = v.tolist()
                f.create_dataset(k, data=v)
