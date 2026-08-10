# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Shape ops."""

from . import concat as concat
from . import dynamic_slicing as dynamic_slicing
from . import frontend as frontend
from . import indexing as indexing
from . import indexing_advanced as indexing_advanced
from . import joining as joining
from . import pad_and_tile as misc
from . import reshape as reshape
from . import scatter as scatter
from . import slicing as slicing
from . import space_batch as space_batch
from . import splitting as splitting
from . import utils as utils
from .concat import Append as Append
from .concat import ColumnStack as ColumnStack
from .concat import Dsplit as Dsplit
from .concat import Dstack as Dstack
from .concat import Hsplit as Hsplit
from .concat import Hstack as Hstack
from .concat import Vsplit as Vsplit
from .concat import Vstack as Vstack
from .frontend import atleast_1d as atleast_1d
from .frontend import atleast_2d as atleast_2d
from .frontend import atleast_3d as atleast_3d
from .frontend import block as block
from .frontend import delete as delete
from .frontend import diag_indices as diag_indices
from .frontend import diag_indices_from as diag_indices_from
from .frontend import diagflat as diagflat
from .frontend import expand_dims as expand_dims
from .frontend import fill_diagonal as fill_diagonal
from .frontend import insert as insert
from .frontend import moveaxis as moveaxis
from .frontend import permute as permute
from .frontend import roll as roll
from .frontend import squeeze as squeeze
from .frontend import swapaxes as swapaxes
from .indexing import take_along_axis as take_along_axis
from .joining import append as append
from .joining import column_stack as column_stack
from .joining import concatenate as concatenate
from .joining import dstack as dstack
from .joining import hstack as hstack
from .joining import stack as stack
from .joining import vstack as vstack
from .reshape import Atleast1d as Atleast1d
from .reshape import Atleast2d as Atleast2d
from .reshape import Atleast3d as Atleast3d
from .reshape import Block as Block
from .reshape import Delete as Delete
from .reshape import Diagflat as Diagflat
from .reshape import DiagIndices as DiagIndices
from .reshape import DiagIndicesFrom as DiagIndicesFrom
from .reshape import ExpandDims as ExpandDims
from .reshape import FillDiagonal as FillDiagonal
from .reshape import Insert as Insert
from .reshape import Moveaxis as Moveaxis
from .reshape import Permute as Permute
from .reshape import Resize as Resize
from .reshape import Roll as Roll
from .reshape import Squeeze as Squeeze
from .reshape import Swapaxes as Swapaxes
from .space_batch import space_to_batch as space_to_batch
from .space_batch import space_to_batch_nd as space_to_batch_nd
from .splitting import array_split as array_split
from .splitting import dsplit as dsplit
from .splitting import hsplit as hsplit
from .splitting import split as split
from .splitting import unstack as unstack
from .splitting import vsplit as vsplit

__all__ = [
    "concat",
    "concatenate",
    "append",
    "column_stack",
    "vstack",
    "hstack",
    "dstack",
    "dynamic_slicing",
    "frontend",
    "indexing",
    "indexing_advanced",
    "joining",
    "misc",
    "permute",
    "reshape",
    "flip",
    "fliplr",
    "flipud",
    "Flip",
    "Fliplr",
    "Flipud",
    "scatter",
    "ExpandDims",
    "Resize",
    "Permute",
    "Block",
    "Delete",
    "DiagIndices",
    "DiagIndicesFrom",
    "Diagflat",
    "FillDiagonal",
    "Insert",
    "Moveaxis",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "Swapaxes",
    "Roll",
    "Append",
    "ColumnStack",
    "Dsplit",
    "Dstack",
    "Hsplit",
    "Hstack",
    "Vsplit",
    "Vstack",
    "Atleast1d",
    "Atleast2d",
    "Atleast3d",
    "Squeeze",
    "expand_dims",
    "permute",
    "block",
    "delete",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "fill_diagonal",
    "insert",
    "append",
    "block",
    "column_stack",
    "delete",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "dsplit",
    "dstack",
    "expand_dims",
    "fill_diagonal",
    "hsplit",
    "hstack",
    "insert",
    "moveaxis",
    "permute",
    "roll",
    "squeeze",
    "swapaxes",
    "vsplit",
    "vstack",
    "swapaxes",
    "roll",
    "atleast_2d",
    "atleast_3d",
    "squeeze",
    "slicing",
    "space_batch",
    "space_to_batch",
    "space_to_batch_nd",
    "splitting",
    "stack",
    "unstack",
    "split",
    "array_split",
    "vsplit",
    "hsplit",
    "dsplit",
    "utils",
]
from .frontend import flip as flip
from .frontend import fliplr as fliplr
from .frontend import flipud as flipud
from .reshape import Flip as Flip
from .reshape import Fliplr as Fliplr
from .reshape import Flipud as Flipud
