"""Dtype resolution and type inference logic."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


def resolve_dtype(res_data: TensorOrScalar, first_tensor: Tensor | None) -> DType | str:
    """Resolve the dtype for the eager result.

    Args:
        res_data (TensorOrScalar): The raw result data.
        first_tensor (Tensor | None): The first input tensor, used as fallback.

    Returns:
        DType | str: The resolved DType.
    """
    if hasattr(res_data, "dtype"):
        dtype_str = str(getattr(res_data, "dtype"))
        if "dtype" in dtype_str:
            m = re.search(r"dtype\('(.*?)'\)", dtype_str)
            if m:
                dtype_str = m.group(1)
        if dtype_str.startswith("dtype"):
            dtype_str = "float32"
        dtype_str = dtype_str.split(".")[-1]
        try:
            return DType(dtype_str)
        except ValueError:
            return DType.Float32
    elif first_tensor is not None:
        return first_tensor.dtype
    return DType.Float32


def resolve_output_dtype_and_device(
    first_tensor: Tensor | None,
    kwargs: dict[str, TensorOrScalar | DType | Device],
) -> tuple[DType | str, Device | str | None]:
    """Resolve output dtype and device based on inputs and kwargs.

    Args:
        first_tensor (Tensor | None): The first input tensor.
        kwargs (dict): The keyword arguments passed to the operation.

    Returns:
        tuple[DType | str, Device | str | None]: A tuple containing the resolved dtype and device.
    """
    out_dtype = None
    if "dtype" in kwargs:
        out_dtype = kwargs["dtype"]
    elif first_tensor is not None:
        out_dtype = first_tensor.dtype
    else:
        out_dtype = DType.Float32
    device = first_tensor.device if first_tensor is not None else None
    return cast("DType | str", out_dtype), device
