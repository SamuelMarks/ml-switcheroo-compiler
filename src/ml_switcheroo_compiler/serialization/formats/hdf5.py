"""HDF5 weight serialization format."""

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver


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
        """
        weights: dict[str, np.ndarray] = {}

        def _visit_func(name: str, node: h5py.Group | h5py.Dataset) -> None:
            """Visit HDF5 nodes and extract datasets.

            Args:
                name (str): The name of the node.
                node (h5py.Group | h5py.Dataset): The HDF5 node (Group or Dataset).
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
        """
        # Validate through schema
        validated = WeightSchema(data=weights_np)

        with h5py.File(filepath, "w") as f:
            for key, value in validated.data.items():
                f.create_dataset(key, data=value)
