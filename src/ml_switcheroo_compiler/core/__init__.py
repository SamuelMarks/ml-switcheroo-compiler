"""Core module for ml-switcheroo."""

from .config import ConfigContext, EagerMode, config
import contextlib
from collections.abc import Iterator
from .assertions import evaluate_assertions, record_assertion, clear_assertions
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
from .sparse_tensor import SparseTensor
from .ragged_tensor import RaggedTensor
from .tensor_array import TensorArray

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
    "SparseTensor",
    "RaggedTensor",
    "TensorArray",
    "TracingError",
    "UnimplementedMathError",
    "clear_assertions",
    "config",
    "evaluate_assertions",
    "record_assertion",
    "backend",
    "get_uid",
]


def image_data_format() -> str:
    return "channels_last"


_uid_dict = {}


def get_uid(prefix: str = "") -> int:
    """Get a unique ID."""
    _uid_dict[prefix] = _uid_dict.get(prefix, 0) + 1
    return _uid_dict[prefix]


def backend() -> str:
    """Return backend."""
    return "numpy"


@contextlib.contextmanager
def name_scope(name: str) -> Iterator[None]:
    """Context manager for name scope."""
    yield
