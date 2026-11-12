"""Implementation of types operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.tensorflow.types`)."""

from __future__ import annotations

import importlib.util
import sys
import typing

from ml_switcheroo_compiler.backends.eager import (
    generic_array,
    generic_asarray,
    generic_item,
    generic_zeros,
)
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_eager_tensor, ensure_float
from ml_switcheroo_compiler.backends.protocols import ensure_mod

if typing.TYPE_CHECKING:
    import tensorflow as tf

_has_tf = sys.modules.get("tensorflow") is not None or ("tensorflow" not in sys.modules and importlib.util.find_spec("tensorflow") is not None)
ensure_mod(sys.modules[__name__]).tf = ensure_mod(importlib.import_module("tensorflow") if _has_tf else None)


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
    shape (tuple): Input `shape` argument (shape).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
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
    data (EagerValue): Input `data` argument (data).
        dtype (EagerValue): Input `dtype` argument (dtype).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_eager_tensor(ensure_callable(generic_array)(tf, data, dtype))


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
    data (EagerValue): Input `data` argument (data).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_eager_tensor(ensure_callable(generic_asarray)(tf, data))


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
    data (EagerValue): Input `data` argument (data).

    Returns:
        float: Computed `float` output.
    """
    return ensure_float(ensure_callable(generic_item)(tf, data))
