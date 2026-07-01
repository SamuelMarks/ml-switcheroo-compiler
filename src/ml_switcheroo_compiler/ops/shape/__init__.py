# pylint: disable=duplicate-code

"""Shape manipulation operations package."""

from ml_switcheroo_compiler.ops.shape.concat import (
    ColumnStack,
    Concatenate,
    Dsplit,
    Dstack,
    Hsplit,
    Hstack,
    RowStack,
    Split,
    Stack,
    Vsplit,
    Vstack,
)
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import DynamicSlice, DynamicUpdateSlice
from ml_switcheroo_compiler.ops.shape.indexing_advanced import (
    ArgSort,
    Assign,
    AssignAdd,
    AssignSub,
    Gather,
    GatherNd,
    Scatter,
    ScatterAdd,
    ScatterNd,
    SearchSorted,
    Select,
    Sort,
    Take,
    TakeAlongAxis,
    TensorScatterUpdate,
    TopK,
    Vdot,
    Where,
)
from ml_switcheroo_compiler.ops.shape.reshape import (
    BroadcastInDim,
    BroadcastTo,
    Expand,
    Flatten,
    Meshgrid,
    Moveaxis,
    Permute,
    Repeat,
    Reshape,
    Resize,
    Roll,
    Squeeze,
    Swapaxes,
    Tile,
    Transpose,
    Tril,
    Triu,
)
from ml_switcheroo_compiler.ops.shape.slicing import Slice, StridedSlice

from .frontend import append as append
from .frontend import argpartition as argpartition
from .frontend import argsort as argsort
from .frontend import argwhere as argwhere
from .frontend import array_split as array_split
from .frontend import atleast_1d as atleast_1d
from .frontend import atleast_2d as atleast_2d
from .frontend import atleast_3d as atleast_3d
from .frontend import broadcast_arrays as broadcast_arrays
from .frontend import broadcast_in_dim as broadcast_in_dim
from .frontend import broadcast_to as broadcast_to
from .frontend import choose as choose
from .frontend import column_stack as column_stack
from .frontend import compress as compress
from .frontend import concatenate as concatenate
from .frontend import delete as delete
from .frontend import diag_indices as diag_indices
from .frontend import diag_indices_from as diag_indices_from
from .frontend import diagflat as diagflat
from .frontend import diagonal as diagonal
from .frontend import diff as diff
from .frontend import digitize as digitize
from .frontend import dsplit as dsplit
from .frontend import dstack as dstack
from .frontend import dynamic_slice as dynamic_slice
from .frontend import dynamic_partition as dynamic_partition
from .frontend import dynamic_stitch as dynamic_stitch
from .frontend import tensor_scatter_sub as tensor_scatter_sub
from .frontend import extract_volume_patches as extract_volume_patches
from .frontend import dynamic_update_slice as dynamic_update_slice
from .frontend import expand as expand
from .frontend import expand_dims as expand_dims
from .frontend import flatten as flatten
from .frontend import unflatten as unflatten
from .frontend import view as view
from .frontend import gather as gather
from .frontend import gather_nd as gather_nd
from .frontend import hsplit as hsplit
from .frontend import hstack as hstack
from .frontend import image_resize as image_resize
from .frontend import meshgrid as meshgrid
from .frontend import moveaxis as moveaxis
from .frontend import pad as pad
from .frontend import partition as partition
from .frontend import permute as permute
from .frontend import repeat as repeat
from .frontend import reshape as reshape
from .frontend import reverse as reverse
from .frontend import roll as roll
from .frontend import scatter as scatter
from .frontend import scatter_add as scatter_add
from .frontend import scatter_nd as scatter_nd
from .frontend import searchsorted as searchsorted
from .frontend import select as select
from .frontend import slice as slice
from .frontend import sort as sort
from .frontend import split as split
from .frontend import squeeze as squeeze
from .frontend import stack as stack
from .frontend import strided_slice as strided_slice
from .frontend import swapaxes as swapaxes
from .frontend import take as take
from .frontend import take_along_axis as take_along_axis
from .frontend import tensor_scatter_add as tensor_scatter_add
from .frontend import tensor_scatter_max as tensor_scatter_max
from .frontend import tensor_scatter_min as tensor_scatter_min
from .frontend import tensor_scatter_update as tensor_scatter_update
from .frontend import tile as tile
from .frontend import top_k as top_k
from .frontend import transpose as transpose
from .frontend import tril as tril
from .frontend import triu as triu
from .frontend import unsqueeze as unsqueeze
from .frontend import unstack as unstack
from .frontend import update_slice as update_slice
from .frontend import vsplit as vsplit
from .frontend import vstack as vstack
from .frontend import where as where
from .frontend import unravel_index as unravel_index
from .frontend import boolean_mask as boolean_mask
from .frontend import invert_permutation as invert_permutation
from .frontend import dynamic_shape as dynamic_shape
from .manipulation import (
    depth_to_space as depth_to_space,
    space_to_depth as space_to_depth,
    space_to_batch as space_to_batch,
    with_space_to_batch as with_space_to_batch,
)

# pylint: disable=duplicate-code


from .indexing import DynamicSliceInDim as DynamicSliceInDim
from .indexing import DynamicUpdateSliceInDim as DynamicUpdateSliceInDim
from .indexing import DynamicIndexInDim as DynamicIndexInDim
from .indexing import DynamicUpdateIndexInDim as DynamicUpdateIndexInDim
from .indexing import SliceInDim as SliceInDim
from .indexing import ScatterApply as ScatterApply
from .indexing import ScatterMax as ScatterMax
from .indexing import ScatterMin as ScatterMin
from .indexing import ScatterMul as ScatterMul

from .indexing import put_along_axis as put_along_axis
from .splitting import old_split as old_split

__all__ = [
    "ArgSort",
    "Assign",
    "AssignAdd",
    "AssignSub",
    "BroadcastInDim",
    "BroadcastTo",
    "ColumnStack",
    "Concatenate",
    "Dsplit",
    "Dstack",
    "DynamicIndexInDim",
    "DynamicSlice",
    "DynamicSliceInDim",
    "DynamicUpdateIndexInDim",
    "DynamicUpdateSlice",
    "DynamicUpdateSliceInDim",
    "Expand",
    "Flatten",
    "Gather",
    "GatherNd",
    "Hsplit",
    "Hstack",
    "Meshgrid",
    "Moveaxis",
    "Permute",
    "Repeat",
    "Reshape",
    "Resize",
    "Roll",
    "RowStack",
    "Scatter",
    "ScatterAdd",
    "ScatterApply",
    "ScatterMax",
    "ScatterMin",
    "ScatterMul",
    "ScatterNd",
    "SearchSorted",
    "Select",
    "Slice",
    "SliceInDim",
    "Sort",
    "Split",
    "Squeeze",
    "Stack",
    "StridedSlice",
    "Swapaxes",
    "Take",
    "TakeAlongAxis",
    "TensorScatterUpdate",
    "Tile",
    "TopK",
    "Transpose",
    "Tril",
    "Triu",
    "Vdot",
    "Vsplit",
    "Vstack",
    "Where",
    "append",
    "argpartition",
    "argsort",
    "argwhere",
    "array_split",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "boolean_mask",
    "broadcast_arrays",
    "broadcast_in_dim",
    "broadcast_to",
    "choose",
    "column_stack",
    "compress",
    "concatenate",
    "delete",
    "depth_to_space",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "diagonal",
    "diff",
    "digitize",
    "dsplit",
    "dstack",
    "dynamic_partition",
    "dynamic_shape",
    "dynamic_slice",
    "dynamic_stitch",
    "dynamic_update_slice",
    "expand",
    "expand_dims",
    "extract_volume_patches",
    "flatten",
    "gather",
    "gather_nd",
    "hsplit",
    "hstack",
    "image_resize",
    "invert_permutation",
    "meshgrid",
    "moveaxis",
    "old_split",
    "pad",
    "partition",
    "permute",
    "put_along_axis",
    "repeat",
    "reshape",
    "reverse",
    "roll",
    "scatter",
    "scatter_add",
    "scatter_nd",
    "searchsorted",
    "select",
    "slice",
    "sort",
    "space_to_batch",
    "space_to_depth",
    "split",
    "squeeze",
    "stack",
    "strided_slice",
    "swapaxes",
    "take",
    "take_along_axis",
    "tensor_scatter_add",
    "tensor_scatter_max",
    "tensor_scatter_min",
    "tensor_scatter_sub",
    "tensor_scatter_update",
    "tile",
    "top_k",
    "transpose",
    "tril",
    "triu",
    "unflatten",
    "unravel_index",
    "unsqueeze",
    "unstack",
    "update_slice",
    "view",
    "vsplit",
    "vstack",
    "where",
    "with_space_to_batch",
]
