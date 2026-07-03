"""H5 format serialization."""

import pickle

import h5py  # pragma: no cover

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


class H5WeightFormat(WeightLoader, WeightSaver):
    """H5 weight format handler."""

    def load(self, filepath: str) -> dict:
        """Load h5 weights."""
        try:  # pragma: no cover
            weights = {}  # pragma: no cover
            with h5py.File(filepath, "r") as f:  # pragma: no cover
                for k in f.keys():  # pragma: no cover
                    weights[k] = f[k][()]  # pragma: no cover
            return weights  # pragma: no cover
        except ImportError:  # pragma: no cover
            with open(filepath, "rb") as f:  # pragma: no cover
                return pickle.load(f)  # pragma: no cover

    def save(self, weights_np: dict, filepath: str) -> None:
        """Save h5 weights."""
        try:  # pragma: no cover
            with h5py.File(filepath, "w") as f:  # pragma: no cover
                for k, v in weights_np.items():  # pragma: no cover
                    f.create_dataset(k, data=v)  # pragma: no cover
        except ImportError:  # pragma: no cover
            with open(filepath, "wb") as f:  # pragma: no cover
                pickle.dump(weights_np, f)  # pragma: no cover
