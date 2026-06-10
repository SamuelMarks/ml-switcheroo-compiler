"""Core module for ml-switcheroo."""

from .dtype import DType, QuantDType
from .device import Device, DeviceType
from .errors import (
    SwitcherooError,
    TracingError,
    CompilationError,
    ShapeMismatchError,
    DTypePromotionError,
    BackendNotSupportedError,
    UnimplementedMathError,
)
from .config import config, ConfigContext, EagerMode
from .tensor import Tensor

__all__ = [
    "DType",
    "QuantDType",
    "Device",
    "DeviceType",
    "SwitcherooError",
    "TracingError",
    "CompilationError",
    "ShapeMismatchError",
    "DTypePromotionError",
    "BackendNotSupportedError",
    "UnimplementedMathError",
    "config",
    "ConfigContext",
    "EagerMode",
    "Tensor",
]
