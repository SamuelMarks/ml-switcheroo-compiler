"""Generic backend-parametric eager reference operations for `reference_math.math_bit`."""

from __future__ import annotations

import types
from collections.abc import Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _get_np_arg  # noqa: F401
from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("Clz")
def _np_clz(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Count the number of leading zero bits in the integer representation of the input.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        TypeError: If invalid inputs or unsupported runtime conditions trigger `TypeError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x_arr = arr_mod.asarray(x)
    if not arr_mod.issubdtype(x_arr.dtype, arr_mod.integer):
        raise TypeError("Clz requires integer inputs.")
    bit_width = int(coerce_int(adapt_backend_module(x_arr).require_symbol("itemsize"))) * 8

    def _clz_scalar(val: BackendArg) -> int:
        """Execute the eager clz scalar operation on the active backend array.

        Args:
        val (BackendArg): Input `val` argument (val).

        Returns:
            int: Computed `int` output.
        """
        val_i: int = int(coerce_int(val))
        if val_i < 0:
            val_i = val_i & (1 << int(bit_width)) - 1
        return int(bit_width) - val_i.bit_length()

    res = arr_mod.vectorize(_clz_scalar)(x_arr)
    return res.astype(x_arr.dtype)


@global_eager_registry.register("BitcastConvertType")
def _np_bitcast_convert_type(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    new_dtype: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Bitcast a tensor from one type to another without changing its underlying memory.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        new_dtype (BackendArg): Input `new_dtype` argument (new dtype).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    dt = adapt_backend_module(arr_mod).get_symbol(str(new_dtype).split(".")[-1], arr_mod.float32)
    _typed_ret_1: BackendArg = arr_mod.asarray(x).view(dt)
    return _typed_ret_1


@global_eager_registry.register("Packbits")
def _np_packbits(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Pack the elements of a binary-valued array into bits in a uint8 array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("packbits")(arr_mod.asarray(args[0]), **kwargs)


@global_eager_registry.register("Unpackbits")
def _np_unpackbits(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Unpack elements of a uint8 array into a binary-valued output array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Variable positional arguments.
        **kwargs (BackendArg): Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("unpackbits")(arr_mod.asarray(args[0]), **kwargs)


@global_eager_registry.register("BitwiseCount")
def _np_bitwise_count(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np bitwise count operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = arr_mod.asarray(args[0])
    return arr_mod.array(
        [bin(bool(n) if type(n).__name__ in ("bool", "bool_") else n).count("1") for n in x.flat]
    ).reshape(x.shape)
