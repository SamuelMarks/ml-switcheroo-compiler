# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Math Ops."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_misc_ext import _get_np_arg, _get_sc


@numpy_eager_registry.register("SegmentSum")
def _np_segment_sum(backend_module: Any, data: Any, segment_ids: Any, num_segments: Any = None, **kwargs: Any) -> Any:
    """Evaluate _np_segment_sum operation.

    Args:
        backend_module (object): The backend_module parameter.
        data (object): The data parameter.
        segment_ids (object): The segment_ids parameter.
        num_segments (object): The num_segments parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    num_segments = num_segments if num_segments is not None else np.max(segment_ids) + 1
    out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    np.add.at(out, segment_ids, data)
    return out


@numpy_eager_registry.register("SparseSegmentSum")
def _np_sparsesegmentsum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseSegmentSum.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    data = backend_module.asarray(args[0])
    return backend_module.sum(data, axis=0, keepdims=True)
