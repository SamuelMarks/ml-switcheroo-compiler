"""Types utils for eager backend."""

from __future__ import annotations

import typing


class ZerosModule(typing.Protocol):
    """Protocol for module providing zeros."""

    def zeros(
        self,
        shape: tuple[int, ...],
    ) -> tuple[str, tuple[int, ...]] | tuple[int, ...]:
        """Create zeros.

        Args:
            shape: The shape parameter.

        Returns:
            Union[tuple[str, tuple[int, ...]], tuple[int, ...]]: Computed result.
        """
        ...


class ArrayModule(typing.Protocol):
    """Protocol for module providing array."""

    def array(
        self,
        data: list[int] | list[float] | tuple[int, ...] | int | float | str,
        dtype: str | type | None = None,
    ) -> tuple[str, list[int] | list[float] | tuple[int, ...] | int | float | str, str | type | None] | list[int] | list[float]:
        """Create array.

        Args:
            data: The data parameter.
            dtype: The dtype parameter.

        Returns:
            Union[tuple[str, Union[list[int], list[float], tuple[int, ...], int, float, str], Union[str, type, None]], list[int], list[float]]: Computed result.
        """
        ...


class AsArrayModule(typing.Protocol):
    """Protocol for module providing asarray."""

    def asarray(
        self,
        data: list[int] | list[float] | tuple[int, ...] | int | float | str,
    ) -> tuple[str, list[int] | list[float] | tuple[int, ...] | int | float | str] | list[int] | list[float]:
        """Create asarray.

        Args:
            data: The data parameter.

        Returns:
            Union[tuple[str, Union[list[int], list[float], tuple[int, ...], int, float, str]], list[int], list[float]]: Computed result.
        """
        ...


class HasItem(typing.Protocol):
    """Protocol for objects providing item method."""

    def item(self) -> float | int:
        """Extract item.

        Returns:
            Union[float, int]: Computed result.
        """
        ...


InputData = typing.Union[list[int], list[float], tuple[int, ...], int, float, "str"]
ArrayResult = typing.Union[tuple[str, InputData, typing.Union[str, type, "None"]], tuple[str, InputData], list[int], "list[float]"]


def generic_zeros(
    mod: ZerosModule,
    shape: tuple[int, ...],
) -> tuple[str, tuple[int, ...]] | tuple[int, ...]:
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
    dtype: str | type | None = None,
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
) -> tuple[str, InputData] | list[int] | list[float]:
    """Provide generic asarray.

    Args:
        mod (AsArrayModule): The module parameter.
        data (InputData): The data parameter.

    Returns:
        tuple[str, InputData] | list[int] | list[float]: Result.
    """
    return mod.asarray(data)


def generic_item(
    mod: ZerosModule | ArrayModule | AsArrayModule | None,
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
