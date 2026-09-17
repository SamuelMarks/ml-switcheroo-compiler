"""HDF5 weight serialization format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import h5py as h5py_types


class WeightSchema(BaseModel):
    """Schema for validating weight structures."""

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    data: dict[str, np.ndarray] = Field(description="The weight data")


class HDF5WeightLoader(WeightLoader):
    """HDF5 implementation for loading weights."""

    def load(self, filepath: str) -> dict[str, np.ndarray]:
        """Load weights from an HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.

        Returns:
            dict[str, np.ndarray]: The loaded weights.

        Raises:
            ImportError: If h5py is not installed.
        """
        if h5py is None:
            raise ImportError("h5py is required for HDF5 weight loading. Please install h5py.")

        weights: dict[str, np.ndarray] = {}

        def _visit_func(name: str, node: h5py_types.Group | h5py_types.Dataset) -> None:
            """Visit HDF5 nodes and extract datasets.

            Args:
                name (str): The name of the node.
                node (h5py_types.Group | h5py_types.Dataset): The HDF5 node (Group or Dataset).
            """
            if isinstance(node, h5py.Dataset):
                weights[name] = node[()]

        with h5py.File(filepath, "r") as f:
            f.visititems(_visit_func)

        # Validate through schema
        validated = WeightSchema(data=weights)
        return validated.data


class HDF5WeightSaver(WeightSaver):
    """HDF5 implementation for saving weights."""

    def save(self, weights_np: dict[str, np.ndarray], filepath: str) -> None:
        """Save weights to an HDF5 file.

        Args:
            weights_np (dict[str, np.ndarray]): The weights to save.
            filepath (str): Path to the HDF5 file.

        Raises:
            ImportError: If h5py is not installed.
        """
        if h5py is None:
            raise ImportError("h5py is required for HDF5 weight saving. Please install h5py.")

        # Validate through schema
        validated = WeightSchema(data=weights_np)

        with h5py.File(filepath, "w") as f:
            for key, value in validated.data.items():
                f.create_dataset(key, data=value)
