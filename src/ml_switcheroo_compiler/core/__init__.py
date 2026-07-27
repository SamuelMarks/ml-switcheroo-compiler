"""Core module for ml-switcheroo."""

from .assertions import clear_assertions, evaluate_assertions, record_assertion
from .config import ConfigContext, EagerMode, config
from .dataset import AudioDataset, Dataset, ImageDataset, NumpyDataset, TextDataset
from .device import Device, DeviceType
from .dtype import DType, QuantDType
from .errors import (
    BackendNotSupportedError,
    CompilationError,
    DTypePromotionError,
    ShapeMismatchError,
    SwitcherooError,
    TracingError,
)
from .ragged_tensor import RaggedTensor
from .sparse_tensor import SparseTensor, SparseTensorCOO, SparseTensorCSR
from .tensor import Tensor
from .tensor_array import TensorArray


def image_data_format() -> str:
    """Evaluate and process the image data format operation.

    Returns:
        str: The evaluated or processed output.
    """
    return "channels_last"


_uid_dict = {}


def get_uid(prefix: str = "") -> int:
    """Get a unique ID."""
    _uid_dict[prefix] = _uid_dict.get(prefix, 0) + 1
    return _uid_dict[prefix]


def backend() -> str:
    """Return backend."""
    return "numpy"


__all__ = [
    "AudioDataset",
    "BackendNotSupportedError",
    "CompilationError",
    "ConfigContext",
    "DType",
    "DTypePromotionError",
    "Dataset",
    "Device",
    "DeviceType",
    "EagerMode",
    "ImageDataset",
    "NumpyDataset",
    "QuantDType",
    "RaggedTensor",
    "ShapeMismatchError",
    "SparseTensor",
    "SparseTensorCOO",
    "SparseTensorCSR",
    "SwitcherooError",
    "Tensor",
    "TensorArray",
    "TextDataset",
    "TracingError",
    "clear_assertions",
    "config",
    "evaluate_assertions",
    "record_assertion",
]
