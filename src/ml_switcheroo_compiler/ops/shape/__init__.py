"""Shape manipulation operations package."""

from ml_switcheroo_compiler.ops.shape.basic import (
    BroadcastInDim,
    BroadcastTo,
    DynamicSlice,
    DynamicUpdateSlice,
    Reshape,
    Resize,
    Sort,
    TopK,
    Transpose,
)

from .frontend import (
    array_split as array_split,
)
from .frontend import (
    broadcast_in_dim as broadcast_in_dim,
)
from .frontend import (
    broadcast_to as broadcast_to,
)
from .frontend import (
    concatenate as concatenate,
)
from .frontend import (
    dsplit as dsplit,
)
from .frontend import (
    dstack as dstack,
)
from .frontend import (
    dynamic_slice as dynamic_slice,
)
from .frontend import (
    dynamic_update_slice as dynamic_update_slice,
)
from .frontend import (
    expand as expand,
)
from .frontend import (
    expand_dims as expand_dims,
)
from .frontend import (
    flatten as flatten,
)
from .frontend import (
    gather as gather,
)
from .frontend import (
    gather_nd as gather_nd,
)
from .frontend import (
    hsplit as hsplit,
)
from .frontend import (
    hstack as hstack,
)
from .frontend import (
    image_resize as image_resize,
)
from .frontend import (
    meshgrid as meshgrid,
)
from .frontend import (
    moveaxis as moveaxis,
)
from .frontend import (
    pad as pad,
)
from .frontend import (
    permute as permute,
)
from .frontend import (
    repeat as repeat,
)
from .frontend import (
    reshape as reshape,
)
from .frontend import (
    roll as roll,
)
from .frontend import (
    scatter as scatter,
)
from .frontend import (
    scatter_add as scatter_add,
)
from .frontend import (
    scatter_nd as scatter_nd,
)
from .frontend import (
    searchsorted as searchsorted,
)
from .frontend import (
    select as select,
)
from .frontend import (
    slice as slice,
)
from .frontend import (
    sort as sort,
)
from .frontend import (
    split as split,
)
from .frontend import (
    squeeze as squeeze,
)
from .frontend import (
    stack as stack,
)
from .frontend import (
    strided_slice as strided_slice,
)
from .frontend import (
    swapaxes as swapaxes,
)
from .frontend import (
    take as take,
)
from .frontend import (
    take_along_axis as take_along_axis,
)
from .frontend import (
    tile as tile,
)
from .frontend import (
    top_k as top_k,
)
from .frontend import (
    transpose as transpose,
)
from .frontend import (
    tril as tril,
)
from .frontend import (
    triu as triu,
)
from .frontend import (
    unsqueeze as unsqueeze,
)
from .frontend import (
    unstack as unstack,
)
from .frontend import (
    update_slice as update_slice,
)
from .frontend import (
    vsplit as vsplit,
)
from .frontend import (
    vstack as vstack,
)
from .frontend import (
    where as where,
)

__all__ = [
    "BroadcastInDim",
    "BroadcastTo",
    "DynamicSlice",
    "DynamicUpdateSlice",
    "Reshape",
    "Resize",
    "Sort",
    "TopK",
    "Transpose",
    "array_split",
    "broadcast_in_dim",
    "broadcast_to",
    "concatenate",
    "dsplit",
    "dstack",
    "dynamic_slice",
    "dynamic_update_slice",
    "expand",
    "flatten",
    "gather",
    "gather_nd",
    "hsplit",
    "hstack",
    "image_resize",
    "meshgrid",
    "moveaxis",
    "pad",
    "permute",
    "repeat",
    "reshape",
    "roll",
    "scatter",
    "scatter_add",
    "scatter_nd",
    "select",
    "slice",
    "sort",
    "split",
    "squeeze",
    "stack",
    "strided_slice",
    "swapaxes",
    "take",
    "take_along_axis",
    "tile",
    "top_k",
    "transpose",
    "tril",
    "triu",
    "unsqueeze",
    "unstack",
    "update_slice",
    "vsplit",
    "vstack",
    "where",
]
