# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""H5 format serialization."""

import h5py

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class H5WeightFormat(WeightLoader, WeightSaver):
    """H5 weight format handler."""

    def load(self, filepath: str):
        """Load h5 weights.

        Args:
        filepath (str): The filepath parameter.

        Returns:
        dict: Result.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        if hasattr(backend, "load_h5"):
            return backend.load_h5(filepath)

        result = {}
        with h5py.File(filepath, "r") as f:

            def _visit(name: str, node) -> None:
                """Visit h5py items to extract datasets.

                Args:
                    name (str): The dataset name.
                    node (object): The node object.
                """
                if isinstance(node, h5py.Dataset):
                    result[name] = node[()]  # h5py returns numpy arrays from slicing

            f.visititems(_visit)
        return result

    def save(self, weights_np, filepath: str) -> None:
        """Save h5 weights.

        Args:
            weights_np (dict): The weights_np parameter.
            filepath (str): The filepath parameter.

        Returns:
            tuple[int, ...]: Result.
        """
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
