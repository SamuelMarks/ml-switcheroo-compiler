"""Shape manipulation operations package."""

from ml_switcheroo.ops.shape.basic import (
    Reshape,
    Transpose,
    BroadcastTo,
)

__all__ = [
    "Reshape",
    "Transpose",
    "BroadcastTo",
]
from .frontend import (
    broadcast_to as broadcast_to,
    concatenate as concatenate,
    dynamic_slice as dynamic_slice,
    expand as expand,
    flatten as flatten,
    gather as gather,
    gather_nd as gather_nd,
    meshgrid as meshgrid,
    moveaxis as moveaxis,
    pad as pad,
    permute as permute,
    repeat as repeat,
    reshape as reshape,
    roll as roll,
    scatter as scatter,
    scatter_add as scatter_add,
    scatter_nd as scatter_nd,
    slice as slice,
    split as split,
    squeeze as squeeze,
    stack as stack,
    strided_slice as strided_slice,
    swapaxes as swapaxes,
    take as take,
    take_along_axis as take_along_axis,
    tile as tile,
    transpose as transpose,
    tril as tril,
    triu as triu,
    unsqueeze as unsqueeze,
    unstack as unstack,
    update_slice as update_slice,
    where as where,
)
