"""Core module for ml-switcheroo."""

from .config import ConfigContext, EagerMode, config
from .dataset import Dataset
from .device import Device, DeviceType
from .dtype import DType, QuantDType
from .errors import (
    BackendNotSupportedError,
    CompilationError,
    DTypePromotionError,
    ShapeMismatchError,
    SwitcherooError,
    TracingError,
    UnimplementedMathError,
)
from .tensor import Tensor

__all__ = [
    "BackendNotSupportedError",
    "CompilationError",
    "ConfigContext",
    "DType",
    "Dataset",
    "DTypePromotionError",
    "Device",
    "DeviceType",
    "EagerMode",
    "QuantDType",
    "ShapeMismatchError",
    "SwitcherooError",
    "Tensor",
    "TracingError",
    "UnimplementedMathError",
    "config",
]
