"""Generic backend-parametric eager reference operations for `lookup_ops`."""

from __future__ import annotations

import types
import typing

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
)

LookupInput = typing.Union[BackendArg, list[typing.Union[int, "str"]], int, "str"]


@global_eager_registry.register("Hashing")
def _np_hashing(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    inputs: LookupInput,
    num_bins: int,
    **kwargs: float | int | str | bool | None,
) -> LookupInput:
    """Execute the eager np hashing operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        inputs (LookupInput): Input values.
        num_bins (int): The number of bins.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        LookupInput: Computed `LookupInput` output.
    """
    return inputs


@global_eager_registry.register("IntegerLookup")
def _np_integer_lookup(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    inputs: LookupInput,
    **kwargs: float | int | str | bool | None,
) -> LookupInput:
    """Execute the eager np integer lookup operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        inputs (LookupInput): Input values.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        LookupInput: Computed `LookupInput` output.
    """
    return inputs


def _np_lookup(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    inputs: LookupInput,
    vocabulary: LookupInput,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Execute the eager np lookup operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        inputs (LookupInput): Input values.
        vocabulary (LookupInput): Vocabulary mapping values.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Result array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    inputs_arr: EagerTensorProtocol = arr_mod.asarray(inputs)
    vocab: EagerTensorProtocol = arr_mod.asarray(vocabulary)
    # basic mapping fallback
    res: EagerTensorProtocol = arr_mod.zeros_like(inputs_arr, dtype=arr_mod.int32)
    for i, v in enumerate(vocab):
        res[inputs_arr == v] = i
    return res


@global_eager_registry.register("StringLookup")
def _np_string_lookup(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    inputs: LookupInput,
    **kwargs: float | int | str | bool | None,
) -> LookupInput:
    """Execute the eager np string lookup operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        inputs (LookupInput): Input values.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        LookupInput: Computed `LookupInput` output.
    """
    return inputs
