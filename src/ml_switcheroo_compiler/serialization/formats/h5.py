"""H5 format serialization."""

from __future__ import annotations

import typing
from types import ModuleType

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver

h5py: ModuleType | None
try:
    import h5py as _h5py

    h5py = _h5py
except ImportError:
    h5py = None

if typing.TYPE_CHECKING:
    import h5py as h5py_types


class H5WeightFormat(WeightLoader, WeightSaver):
    """H5 weight format handler."""

    def load(self, filepath: str) -> dict[str, TensorLike]:
        """Load h5 weights.

        Args:
            filepath (str): The filepath parameter.

        Returns:
            dict[str, TensorLike]: Loaded weights dictionary.

        Raises:
            ImportError: If h5py is not installed.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        if hasattr(backend, "load_h5"):
            return typing.cast(dict[str, TensorLike], backend.load_h5(filepath))

        if h5py is None:
            raise ImportError("h5py is required for H5 weight loading. Please install h5py.")

        np_mod = BackendRegistry.get("numpy").get_module()
        result: dict[str, TensorLike] = {}
        with h5py.File(filepath, "r") as f:

            def _visit(name: str, node: h5py_types.Group | h5py_types.Dataset) -> None:
                """Visit h5py items to extract datasets.

                Args:
                    name (str): The dataset name.
                    node (h5py_types.Group | h5py_types.Dataset): The node object.
                """
                if isinstance(node, h5py.Dataset):
                    result[name] = np_mod.asarray(node[()])

            f.visititems(_visit)
        return result

    def save(self, weights_np: dict[str, TensorLike], filepath: str) -> None:
        """Save h5 weights.

        Args:
            weights_np (dict[str, TensorLike]): The weights dictionary to save.
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
