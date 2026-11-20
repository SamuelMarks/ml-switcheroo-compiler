"""Serialization formats package."""

from __future__ import annotations

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver
from ml_switcheroo_compiler.serialization.formats.h5 import (
    H5WeightFormat,
    HDF5WeightLoader,
    HDF5WeightSaver,
    WeightSchema,
)

__all__ = ["H5WeightFormat", "HDF5WeightLoader", "HDF5WeightSaver", "WeightLoader", "WeightSaver", "WeightSchema"]
