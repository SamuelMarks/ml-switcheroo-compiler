"""Dtype resolution and type inference logic."""

import re
from ml_switcheroo_compiler.core.dtype import DType


def resolve_dtype(res_data: object, first_tensor: object) -> object:
    """Resolve the dtype for the eager result."""
    if hasattr(res_data, "dtype"):
        dtype_str = str(res_data.dtype)
        if "dtype" in dtype_str:  # pragma: no branch
            m = re.search(r"dtype\('(.*?)'\)", dtype_str)  # pragma: no cover
            if m:  # pragma: no cover
                dtype_str = m.group(1)  # pragma: no cover
        if dtype_str.startswith("dtype"):  # pragma: no branch
            dtype_str = "float32"  # pragma: no cover
        dtype_str = dtype_str.split(".")[-1]
        try:
            return DType(dtype_str)
        except ValueError:
            return DType.Float32
    elif first_tensor is not None:
        return first_tensor.dtype
    return DType.Float32


def resolve_output_dtype_and_device(first_tensor: object, kwargs: dict) -> tuple[object, object]:
    """Resolve output dtype and device."""
    out_dtype = None
    if "dtype" in kwargs:
        out_dtype = kwargs["dtype"]
    elif first_tensor is not None:
        out_dtype = first_tensor.dtype
    else:
        out_dtype = DType.Float32
    device = first_tensor.device if first_tensor is not None else None
    return out_dtype, device
