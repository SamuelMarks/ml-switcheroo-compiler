# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

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
    """Evaluate image_data_format operation.

    Returns:
        str: Result.
    """
    return "channels_last"


_uid_dict = {}  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


def get_uid(prefix: str = "") -> int:
    """Get a unique ID for a prefix.

    Args:
        prefix (str): Prefix.

    Returns:
        int: uid.
    """
    _uid_dict[prefix] = _uid_dict.get(prefix, 0) + 1
    return _uid_dict[prefix]


def backend() -> str:
    """Return backend name.

    Returns:
        str: name.
    """
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
