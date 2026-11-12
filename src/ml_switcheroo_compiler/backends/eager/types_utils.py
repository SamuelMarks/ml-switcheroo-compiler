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
            shape: Input `shape` argument (shape).

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
            data: Input `data` argument (data).
            dtype: Input `dtype` argument (dtype).

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
            data: Input `data` argument (data).

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
        mod (ZerosModule): Input `module` argument (module).
        shape (tuple[int, ...]): Input `shape` argument (shape).

    Returns:
        tuple[str, tuple[int, ...]] | tuple[int, ...]: Computed `tuple[str, tuple[int, ...]] | tuple[int, ...]` output.
    """
    return mod.zeros(shape)


def generic_array(
    mod: ArrayModule,
    data: InputData,
    dtype: str | type | None = None,
) -> ArrayResult:
    """Provide generic array.

    Args:
        mod (ArrayModule): Input `module` argument (module).
        data (InputData): Input `data` argument (data).
        dtype (Union[str, type, None]): Input `dtype` argument (dtype).

    Returns:
        ArrayResult: Computed `ArrayResult` output.
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
        mod (AsArrayModule): Input `module` argument (module).
        data (InputData): Input `data` argument (data).

    Returns:
        tuple[str, InputData] | list[int] | list[float]: Computed `tuple[str, InputData] | list[int] | list[float]` output.
    """
    return mod.asarray(data)


def generic_item(
    mod: ZerosModule | ArrayModule | AsArrayModule | None,
    data: HasItem,
) -> float:
    """Provide generic item.

    Args:
        mod (ZerosModule | ArrayModule | AsArrayModule | None): Input `module` argument (module).
        data (HasItem): Input `data` argument (data).

    Returns:
        float: Computed `float` output.
    """
    return float(data.item())
