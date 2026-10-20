"""Serialization formats package."""

from __future__ import annotations

from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver
from ml_switcheroo_compiler.serialization.formats.hdf5 import HDF5WeightLoader, HDF5WeightSaver

__all__ = ["WeightLoader", "WeightSaver", "HDF5WeightLoader", "HDF5WeightSaver"]
