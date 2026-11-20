"""Padding, tiling, sorting, and miscellaneous shape/query operations facade."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend as get_active_backend
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    DynamicShape as DynamicShape,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    Pad as Pad,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    Rank as Rank,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    Size as Size,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    _compute_pad_dim as _compute_pad_dim,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    _normalize_pad_width as _normalize_pad_width,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    image_resize as image_resize,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    pad as pad,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    pad_circular as pad_circular,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    pad_constant as pad_constant,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    pad_reflect as pad_reflect,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_padding import (
    pad_replicate as pad_replicate,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    Flatnonzero as Flatnonzero,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    Nonzero as Nonzero,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    Percentile as Percentile,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    Quantile as Quantile,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    RavelMultiIndex as RavelMultiIndex,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    Unique as Unique,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    _get_unique_inverse_shape as _get_unique_inverse_shape,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    _infer_shape_percentile_quantile as _infer_shape_percentile_quantile,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    flatnonzero as flatnonzero,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    nonzero as nonzero,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    percentile as percentile,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    quantile as quantile,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    ravel_multi_index as ravel_multi_index,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_query import (
    unique as unique,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    Repeat as Repeat,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    Tile as Tile,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    _compute_meshgrid_shape as _compute_meshgrid_shape,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    _compute_tile_shape as _compute_tile_shape,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    _repeat_infer_no_axis as _repeat_infer_no_axis,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    meshgrid as meshgrid,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    repeat as repeat,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    tile as tile,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    tril as tril,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_repeat import (
    triu as triu,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_sorting import (
    Lexsort as Lexsort,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_sorting import (
    Searchsorted as Searchsorted,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_sorting import (
    SortComplex as SortComplex,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_sorting import (
    argsort as argsort,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_sorting import (
    lexsort as lexsort,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_sorting import (
    searchsorted as searchsorted,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_sorting import (
    sort as sort,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_sorting import (
    sort_complex as sort_complex,
)
from ml_switcheroo_compiler.ops.shape._pad_tile_sorting import (
    top_k as top_k,
)
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node as _emit_shape_node

__all__ = [
    "DynamicShape",
    "Flatnonzero",
    "Lexsort",
    "Nonzero",
    "Pad",
    "Percentile",
    "Quantile",
    "Rank",
    "RavelMultiIndex",
    "Repeat",
    "Searchsorted",
    "Size",
    "SortComplex",
    "Tile",
    "Unique",
    "_compute_meshgrid_shape",
    "_compute_pad_dim",
    "_compute_tile_shape",
    "_emit_shape_node",
    "_get_unique_inverse_shape",
    "_infer_shape_percentile_quantile",
    "_normalize_pad_width",
    "_repeat_infer_no_axis",
    "argsort",
    "flatnonzero",
    "get_active_backend",
    "image_resize",
    "lexsort",
    "meshgrid",
    "nonzero",
    "pad",
    "pad_circular",
    "pad_constant",
    "pad_reflect",
    "pad_replicate",
    "percentile",
    "quantile",
    "ravel_multi_index",
    "repeat",
    "searchsorted",
    "sort",
    "sort_complex",
    "tile",
    "top_k",
    "tril",
    "triu",
    "unique",
]
