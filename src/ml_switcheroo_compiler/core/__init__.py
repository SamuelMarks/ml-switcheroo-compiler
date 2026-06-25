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
from .sparse_tensor import SparseTensor, SparseTensorCOO, SparseTensorCSR
from .ragged_tensor import RaggedTensor
from .tensor_array import TensorArray


def image_data_format() -> str:
    """Function docstring."""
    return "channels_last"  # pragma: no cover


_uid_dict = {}


def get_uid(prefix: str = "") -> int:
    """Get a unique ID."""
    _uid_dict[prefix] = _uid_dict.get(prefix, 0) + 1  # pragma: no cover
    return _uid_dict[prefix]  # pragma: no cover


def backend() -> str:
    """Return backend."""
    return "numpy"  # pragma: no cover


@contextlib.contextmanager
def name_scope(name: str) -> Iterator[None]:
    """Context manager for name scope."""
    yield  # pragma: no cover


__all__ = [
    "BackendNotSupportedError",
    "CompilationError",
    "ConfigContext",
    "DType",
    "DTypePromotionError",
    "Dataset",
    "Device",
    "DeviceType",
    "EagerMode",
    "QuantDType",
    "RaggedTensor",
    "ShapeMismatchError",
    "SparseTensor",
    "SparseTensorCOO",
    "SparseTensorCSR",
    "SwitcherooError",
    "Tensor",
    "TensorArray",
    "TracingError",
    "UnimplementedMathError",
    "backend",
    "clear_assertions",
    "config",
    "evaluate_assertions",
    "get_uid",
    "record_assertion",
]
