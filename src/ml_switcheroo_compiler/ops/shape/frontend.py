"""Defines shape, memory, and movement operations for Tensor objects.

This module provides functions to manipulate tensor shapes, dimensions, and memory
layouts, supporting both eager execution (using NumPy) and lazy execution (by tracing
and emitting logical nodes to a graph)
"""

from ml_switcheroo_compiler.ops.shape.indexing import (
    gather,
    gather_nd,
    scatter,
    scatter_add,
    scatter_nd,
    searchsorted,
    select,
    take,
    take_along_axis,
    where,
)
from ml_switcheroo_compiler.ops.shape.joining import concatenate, dstack, hstack, stack, vstack
from ml_switcheroo_compiler.ops.shape.manipulation import (
    broadcast_in_dim,
    broadcast_to,
    expand,
    expand_dims,
    flatten,
    moveaxis,
    permute,
    reshape,
    roll,
    squeeze,
    swapaxes,
    transpose,
    unsqueeze,
)
from ml_switcheroo_compiler.ops.shape.misc import (
    image_resize,
    meshgrid,
    pad,
    repeat,
    sort,
    tile,
    top_k,
    tril,
    triu,
)
from ml_switcheroo_compiler.ops.shape.slicing import (
    dynamic_slice,
    dynamic_update_slice,
    slice,
    strided_slice,
    update_slice,
)
from ml_switcheroo_compiler.ops.shape.splitting import (
    array_split,
    dsplit,
    hsplit,
    split,
    unstack,
    vsplit,
)
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

__all__ = [
    "_emit_shape_node",
    "array_split",
    "broadcast_in_dim",
    "broadcast_to",
    "concatenate",
    "dsplit",
    "dstack",
    "dynamic_slice",
    "dynamic_update_slice",
    "expand",
    "expand_dims",
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
    "searchsorted",
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
