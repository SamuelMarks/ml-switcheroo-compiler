"""Numpy lookup operations."""

from __future__ import annotations

from types import ModuleType
from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

LookupInput = Union[np.ndarray, list[Union[int, str]], int, str]


@numpy_eager_registry.register("Hashing")
def _np_hashing(
    backend_module: ModuleType,
    inputs: LookupInput,
    num_bins: int,
    **kwargs: Union[float, int, str, bool, None],
) -> LookupInput:
    """Evaluate _np_hashing operation.

    Args:
        backend_module (ModuleType): The backend module.
        inputs (LookupInput): Input values.
        num_bins (int): The number of bins.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        LookupInput: Result.
    """
    return inputs


@numpy_eager_registry.register("IntegerLookup")
def _np_integer_lookup(
    backend_module: ModuleType,
    inputs: LookupInput,
    **kwargs: Union[float, int, str, bool, None],
) -> LookupInput:
    """Evaluate _np_integer_lookup operation.

    Args:
        backend_module (ModuleType): The backend module.
        inputs (LookupInput): Input values.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        LookupInput: Result.
    """
    return inputs


@numpy_eager_registry.register("Lookup")
def _np_lookup(
    backend_module: ModuleType,
    inputs: LookupInput,
    vocabulary: LookupInput,
    **kwargs: Union[float, int, str, bool, None],
) -> np.ndarray:
    """Evaluate _np_lookup operation.

    Args:
        backend_module (ModuleType): The backend module.
        inputs (LookupInput): Input values.
        vocabulary (LookupInput): Vocabulary mapping values.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    inputs_arr: np.ndarray = np.asarray(inputs)
    vocab: np.ndarray = np.asarray(vocabulary)
    # basic mapping fallback
    res: np.ndarray = np.zeros_like(inputs_arr, dtype=np.int32)
    for i, v in enumerate(vocab):
        res[inputs_arr == v] = i
    return res


@numpy_eager_registry.register("StringLookup")
def _np_string_lookup(
    backend_module: ModuleType,
    inputs: LookupInput,
    **kwargs: Union[float, int, str, bool, None],
) -> LookupInput:
    """Evaluate _np_string_lookup operation.

    Args:
        backend_module (ModuleType): The backend module.
        inputs (LookupInput): Input values.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        LookupInput: Result.
    """
    return inputs
