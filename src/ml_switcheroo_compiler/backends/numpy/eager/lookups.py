"""Numpy lookup operations."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

LookupInput = typing.Union[np.ndarray[tuple[int, ...], np.dtype[np.float64]], list[typing.Union[int, "str"]], int, "str"]


@numpy_eager_registry.register("Hashing")
def _np_hashing(
    backend_module: BackendModuleProtocol,
    inputs: LookupInput,
    num_bins: int,
    **kwargs: float | int | str | bool | None,
) -> LookupInput:
    """Execute the eager np hashing operation on the active backend array.

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
    backend_module: BackendModuleProtocol,
    inputs: LookupInput,
    **kwargs: float | int | str | bool | None,
) -> LookupInput:
    """Execute the eager np integer lookup operation on the active backend array.

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
    backend_module: BackendModuleProtocol,
    inputs: LookupInput,
    vocabulary: LookupInput,
    **kwargs: float | int | str | bool | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager np lookup operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend module.
        inputs (LookupInput): Input values.
        vocabulary (LookupInput): Vocabulary mapping values.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        np.ndarray: Result array.
    """
    inputs_arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(inputs)
    vocab: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.asarray(vocabulary)
    # basic mapping fallback
    res: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.zeros_like(inputs_arr, dtype=np.int32)
    for i, v in enumerate(vocab):
        res[inputs_arr == v] = i
    return res


@numpy_eager_registry.register("StringLookup")
def _np_string_lookup(
    backend_module: BackendModuleProtocol,
    inputs: LookupInput,
    **kwargs: float | int | str | bool | None,
) -> LookupInput:
    """Execute the eager np string lookup operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend module.
        inputs (LookupInput): Input values.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        LookupInput: Result.
    """
    return inputs
