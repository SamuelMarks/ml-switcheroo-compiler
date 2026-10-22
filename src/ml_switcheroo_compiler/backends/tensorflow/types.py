"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.tensorflow.types`)."""

from __future__ import annotations

import importlib.util
import sys
import typing

if typing.TYPE_CHECKING:
    import tensorflow as tf
else:
    _has_tf = sys.modules.get("tensorflow") is not None or ("tensorflow" not in sys.modules and importlib.util.find_spec("tensorflow") is not None)
    tf = importlib.import_module("tensorflow") if _has_tf else None

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_callable, as_float, as_tensor


def zeros(cls: type, shape: tuple[int, ...]) -> EagerValue:
    """Execute zeros for compiler processing.

    Args:
        cls (type): Class.
        shape (tuple): Shape.

    Args:
        cls (type): Class.
        shape (tuple): Shape.

    Args:
        cls (type): The class.
    shape (tuple): The shape parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return generic_zeros(tf, shape)


def array(cls: type, data: EagerValue, dtype: EagerValue = None) -> EagerValue:
    """Execute array for compiler processing.

    Args:
        cls (type): Class.
        data (EagerValue): Data.
        dtype (EagerValue): Dtype.

    Args:
        cls (type): Class.
        data (EagerValue): Data.
        dtype (EagerValue): Dtype.

    Args:
        cls (type): The class.
    data (EagerValue): The data parameter.
        dtype (EagerValue): The dtype parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return as_tensor(as_callable(generic_array)(tf, data, dtype))


def asarray(cls: type, data: EagerValue) -> EagerValue:
    """Execute asarray for compiler processing.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): The class.
    data (EagerValue): The data parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return as_tensor(as_callable(generic_asarray)(tf, data))


def item(cls: type, data: EagerValue) -> float:
    """Execute item for compiler processing.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): Class.
        data (EagerValue): Data.

    Args:
        cls (type): The class.
    data (EagerValue): The data parameter.

    Returns:
        float: Result.
    """
    return as_float(as_callable(generic_item)(tf, data))
