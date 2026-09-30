"""Types utils for eager backend."""

from __future__ import annotations

from typing import Protocol, Union


class ZerosModule(Protocol):
    """Protocol for module providing zeros."""

    def zeros(
        self,
        shape: tuple[int, ...],
    ) -> Union[tuple[str, tuple[int, ...]], tuple[int, ...]]:
        """Create zeros."""
        ...


class ArrayModule(Protocol):
    """Protocol for module providing array."""

    def array(
        self,
        data: Union[list[int], list[float], tuple[int, ...], int, float, str],
        dtype: Union[str, type, None] = None,
    ) -> Union[
        tuple[
            str,
            Union[list[int], list[float], tuple[int, ...], int, float, str],
            Union[str, type, None],
        ],
        list[int],
        list[float],
    ]:
        """Create array."""
        ...


class AsArrayModule(Protocol):
    """Protocol for module providing asarray."""

    def asarray(
        self,
        data: Union[list[int], list[float], tuple[int, ...], int, float, str],
    ) -> Union[
        tuple[
            str,
            Union[list[int], list[float], tuple[int, ...], int, float, str],
        ],
        list[int],
        list[float],
    ]:
        """Create asarray."""
        ...


class HasItem(Protocol):
    """Protocol for objects providing item method."""

    def item(self) -> Union[float, int]:
        """Extract item."""
        ...


InputData = Union[list[int], list[float], tuple[int, ...], int, float, str]
ArrayResult = Union[
    tuple[str, InputData, Union[str, type, None]],
    tuple[str, InputData],
    list[int],
    list[float],
]


def generic_zeros(
    mod: ZerosModule,
    shape: tuple[int, ...],
) -> Union[tuple[str, tuple[int, ...]], tuple[int, ...]]:
    """Provide generic zeros.

    Args:
        mod (ZerosModule): The module parameter.
        shape (tuple[int, ...]): The shape parameter.

    Returns:
        tuple[str, tuple[int, ...]] | tuple[int, ...]: Result.
    """
    return mod.zeros(shape)


def generic_array(
    mod: ArrayModule,
    data: InputData,
    dtype: Union[str, type, None] = None,
) -> ArrayResult:
    """Provide generic array.

    Args:
        mod (ArrayModule): The module parameter.
        data (InputData): The data parameter.
        dtype (Union[str, type, None]): The dtype parameter.

    Returns:
        ArrayResult: Result.
    """
    if dtype is not None:
        return mod.array(data, dtype=dtype)
    return mod.array(data)


def generic_asarray(
    mod: AsArrayModule,
    data: InputData,
) -> Union[tuple[str, InputData], list[int], list[float]]:
    """Provide generic asarray.

    Args:
        mod (AsArrayModule): The module parameter.
        data (InputData): The data parameter.

    Returns:
        tuple[str, InputData] | list[int] | list[float]: Result.
    """
    return mod.asarray(data)


def generic_item(
    mod: Union[ZerosModule, ArrayModule, AsArrayModule, None],
    data: HasItem,
) -> float:
    """Provide generic item.

    Args:
        mod (ZerosModule | ArrayModule | AsArrayModule | None): The module parameter.
        data (HasItem): The data parameter.

    Returns:
        float: Result.
    """
    return float(data.item())
