# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""H5 format serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import h5py as h5py_types


class H5WeightFormat(WeightLoader, WeightSaver):
    """H5 weight format handler."""

    def load(self, filepath: str) -> dict[str, np.ndarray]:
        """Load h5 weights.

        Args:
            filepath (str): The filepath parameter.

        Returns:
            dict[str, np.ndarray]: Loaded weights dictionary.

        Raises:
            ImportError: If h5py is not installed.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        if hasattr(backend, "load_h5"):
            return backend.load_h5(filepath)

        if h5py is None:
            raise ImportError("h5py is required for H5 weight loading. Please install h5py.")

        result: dict[str, np.ndarray] = {}
        with h5py.File(filepath, "r") as f:

            def _visit(name: str, node: h5py_types.Group | h5py_types.Dataset) -> None:
                """Visit h5py items to extract datasets.

                Args:
                    name (str): The dataset name.
                    node (h5py_types.Group | h5py_types.Dataset): The node object.
                """
                if isinstance(node, h5py.Dataset):
                    result[name] = np.asarray(node[()])

            f.visititems(_visit)
        return result

    def save(self, weights_np: dict[str, np.ndarray], filepath: str) -> None:
        """Save h5 weights.

        Args:
            weights_np (dict[str, np.ndarray]): The weights dictionary to save.
            filepath (str): The destination filepath.

        Raises:
            ImportError: If h5py is not installed.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        if hasattr(backend, "save_h5"):
            backend.save_h5(weights_np, filepath)
            return

        if h5py is None:
            raise ImportError("h5py is required for H5 weight saving. Please install h5py.")

        with h5py.File(filepath, "w") as f:
            for k, v in weights_np.items():
                if hasattr(v, "numpy"):
                    v = v.numpy()
                elif hasattr(v, "data") and hasattr(v.data, "numpy"):
                    v = v.data.numpy()
                elif hasattr(v, "tolist"):
                    v = v.tolist()
                f.create_dataset(k, data=v)
