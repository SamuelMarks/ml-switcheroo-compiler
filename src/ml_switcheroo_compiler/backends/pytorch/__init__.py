"""PyTorch Code Generator and FX Graph Ingestion Package."""

from __future__ import annotations

import importlib.util
import sys

try:
    _has_pkg = importlib.util.find_spec("torch") is not None
except ValueError:
    _has_pkg = True

if not _has_pkg and "sphinx" not in sys.modules and "pytest" not in sys.modules:
    raise ImportError("The 'pytorch' backend requires the 'torch' library to be installed.")

from .distributed_collectives import (
    pytorch_all_gather,
    pytorch_all_reduce,
    pytorch_broadcast,
    pytorch_reduce_scatter,
)
from .eager import execute_op
from .fx_ingest import ingest_torch_fx
from .generator import (
    PyTorchCodeGenerator,
)
from .generator import (
    _pt_all_gather as _all_gather,
)
from .generator import (
    _pt_all_reduce as _all_reduce,
)
from .generator import (
    _pt_broadcast as _broadcast,
)
from .generator import (
    _pt_reduce_scatter as _reduce_scatter,
)
from .types import array, asarray, item, zeros

_ = (_all_gather, _all_reduce, _broadcast, _reduce_scatter)

__all__ = [
    "PyTorchCodeGenerator",
    "array",
    "asarray",
    "execute_op",
    "ingest_torch_fx",
    "item",
    "pytorch_all_gather",
    "pytorch_all_reduce",
    "pytorch_broadcast",
    "pytorch_reduce_scatter",
    "zeros",
]
