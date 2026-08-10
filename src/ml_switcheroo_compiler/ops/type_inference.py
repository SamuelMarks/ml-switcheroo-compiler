# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Dtype resolution and type inference logic."""

import re
from typing import Any

from ml_switcheroo_compiler.core.dtype import DType


def resolve_dtype(res_data: Any, first_tensor: Any) -> Any:
    """Resolve the dtype for the eager result.

    Args:
        res_data (object): The raw result data.
        first_tensor (object): The first input tensor, used as fallback.

    Returns: Any: The resolved DType.
    """
    if hasattr(res_data, "dtype"):
        dtype_str = str(res_data.dtype)
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


def resolve_output_dtype_and_device(first_tensor: Any, kwargs: dict) -> tuple[Any, Any]:
    """Resolve output dtype and device based on inputs and kwargs.

    Args:
        first_tensor (object): The first input tensor.
        kwargs (dict): The keyword arguments passed to the operation.

    Returns:
        tuple[Any, Any]: A tuple containing the resolved dtype and device.
    """
    out_dtype = None
    if "dtype" in kwargs:
        out_dtype = kwargs["dtype"]
    elif first_tensor is not None:
        out_dtype = first_tensor.dtype
    else:
        out_dtype = DType.Float32
    device = first_tensor.device if first_tensor is not None else None
    return out_dtype, device
