# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Dtype resolution and type inference logic."""

import re

from ml_switcheroo_compiler.core.dtype import DType


def resolve_dtype(res_data: object, first_tensor: object) -> object:
    """Resolve the dtype for the eager result.

    Args:
        res_data (object): The raw result data.
        first_tensor (object): The first input tensor, used as fallback.

    Returns: object: The resolved DType.
    """
    if hasattr(res_data, "dtype"):
        dtype_str: object = str(res_data.dtype)
        if "dtype" in dtype_str:
            m: object = re.search(r"dtype\('(.*?)'\)", dtype_str)
            if m:
                dtype_str: object = m.group(1)
        if dtype_str.startswith("dtype"):
            dtype_str: object = "float32"
        dtype_str: object = dtype_str.split(".")[-1]
        try:
            return DType(dtype_str)
        except ValueError:
            return DType.Float32
    elif first_tensor is not None:
        return first_tensor.dtype
    return DType.Float32


def resolve_output_dtype_and_device(first_tensor: object, kwargs: dict[str, object]) -> tuple[object, object]:
    """Resolve output dtype and device based on inputs and kwargs.

    Args:
        first_tensor (object): The first input tensor.
        kwargs (dict): The keyword arguments passed to the operation.

    Returns:
        tuple[object, object]: A tuple containing the resolved dtype and device.
    """
    out_dtype: object = None
    if "dtype" in kwargs:
        out_dtype: object = kwargs["dtype"]
    elif first_tensor is not None:
        out_dtype: object = first_tensor.dtype
    else:
        out_dtype: object = DType.Float32
    device: object = first_tensor.device if first_tensor is not None else None
    return out_dtype, device
