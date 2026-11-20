"""Types utils for eager backend."""

from __future__ import annotations

import types
import typing

from ml_switcheroo_compiler.core.shape import OpValue


@typing.runtime_checkable
class ZerosModule(typing.Protocol):
    """Protocol for module providing zeros."""

    def zeros(
        self,
        shape: tuple[int, ...],
    ) -> OpValue:
        """Create zeros.

        Args:
            shape (tuple[int, ...]): Input `shape` argument (shape).

        Returns:
            OpValue: Computed result.
        """
        ...


@typing.runtime_checkable
class ArrayModule(typing.Protocol):
    """Protocol for module providing array."""

    def array(
        self,
        data: OpValue,
        dtype: OpValue = None,
    ) -> OpValue:
        """Create array.

        Args:
            data (OpValue): Input `data` argument (data).
            dtype (OpValue): Input `dtype` argument (dtype).

        Returns:
            OpValue: Computed result.
        """
        ...


@typing.runtime_checkable
class AsArrayModule(typing.Protocol):
    """Protocol for module providing asarray."""

    def asarray(
        self,
        data: OpValue,
    ) -> OpValue:
        """Create asarray.

        Args:
            data (OpValue): Input `data` argument (data).

        Returns:
            OpValue: Computed result.
        """
        ...


@typing.runtime_checkable
class HasItem(typing.Protocol):
    """Protocol for objects providing item method."""

    def item(self) -> float | int:
        """Extract item.

        Returns:
            float | int: Computed result.
        """
        ...


InputData = typing.Union[list[int], list[float], tuple[int, ...], int, float, "str"]

ArrayResult = typing.Union[
    tuple[str, InputData, typing.Union[str, type, "None"]], tuple[str, InputData], list[int], "list[float]"
]


def generic_zeros(
    mod: types.ModuleType | ZerosModule | OpValue,
    shape: tuple[int, ...],
) -> OpValue:
    """Provide generic zeros.

    Args:
        mod (types.ModuleType | ZerosModule | OpValue): Input `module` argument (module).
        shape (tuple[int, ...]): Input `shape` argument (shape).

    Returns:
        OpValue: Computed zeros array.
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    if isinstance(mod, ZerosModule) and not isinstance(mod, types.ModuleType):
        return mod.zeros(shape)
    return adapt_backend_module(mod).zeros(shape)


def generic_array(
    mod: types.ModuleType | ArrayModule | OpValue,
    data: OpValue,
    dtype: OpValue = None,
) -> OpValue:
    """Provide generic array.

    Args:
        mod (types.ModuleType | ArrayModule | OpValue): Input `module` argument (module).
        data (OpValue): Input `data` argument (data).
        dtype (OpValue): Input `dtype` argument (dtype).

    Returns:
        OpValue: Computed array result.
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    if isinstance(mod, ArrayModule) and not isinstance(mod, types.ModuleType):
        if dtype is not None:
            return mod.array(data, dtype=dtype)
        return mod.array(data)
    fn = adapt_backend_module(mod).require_op("array")
    return fn(data, dtype=dtype) if dtype is not None else fn(data)


def generic_asarray(
    mod: types.ModuleType | AsArrayModule | OpValue,
    data: OpValue,
) -> OpValue:
    """Provide generic asarray.

    Args:
        mod (types.ModuleType | AsArrayModule | OpValue): Input `module` argument (module).
        data (OpValue): Input `data` argument (data).

    Returns:
        OpValue: Computed array result.
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    if isinstance(mod, AsArrayModule) and not isinstance(mod, types.ModuleType):
        return mod.asarray(data)
    return adapt_backend_module(mod).asarray(data)


def generic_item(
    mod: types.ModuleType | ZerosModule | ArrayModule | AsArrayModule | OpValue,
    data: HasItem | OpValue,
) -> float:
    """Provide generic item.

    Args:
        mod (types.ModuleType | ZerosModule | ArrayModule | AsArrayModule | OpValue): Input `module` argument (module).
        data (HasItem | OpValue): Input `data` argument (data).

    Returns:
        float: Extracted scalar float.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_float

    _ = mod
    if isinstance(data, HasItem):
        return float(data.item())
    return coerce_float(data)
