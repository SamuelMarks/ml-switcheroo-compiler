# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_testing module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


@numpy_eager_registry.register("Piecewise")
def _np_piecewise(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_piecewise operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return np.piecewise(np.asarray(args[0]), args[1], args[2], *args[3:], **kwargs)


@numpy_eager_registry.register("PromoteTypes")
def _np_promotetypes(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the data type with the smallest size and smallest scalar kind to which both given types can be safely cast.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.promote_types(args[0], args[1])


@numpy_eager_registry.register("ApplyAlongAxis")
def _np_apply_along_axis(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply a function to 1-D slices along the given axis.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.apply_along_axis(*args, **kwargs)


@numpy_eager_registry.register("ArrayEquiv")
def _np_array_equiv_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ArrayEquiv via array_equiv.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.array_equiv(*args, **kwargs)


@numpy_eager_registry.register("BroadcastArrays")
def _np_broadcast_arrays_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement BroadcastArrays via broadcast_arrays.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.broadcast_arrays(*args, **kwargs)


@numpy_eager_registry.register("CanCast")
def _np_can_cast_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement CanCast via can_cast.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.can_cast(*args, **kwargs)


@numpy_eager_registry.register("Histogram")
def _np_histogram_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Histogram via histogram.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.histogram(*args, **kwargs)


@numpy_eager_registry.register("Histogram2d")
def _np_histogram2d_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Histogram2d via histogram2d.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.histogram2d(*args, **kwargs)


@numpy_eager_registry.register("HistogramBinEdges")
def _np_histogram_bin_edges_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement HistogramBinEdges via histogram_bin_edges.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.histogram_bin_edges(*args, **kwargs)


@numpy_eager_registry.register("Histogramdd")
def _np_histogramdd_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Histogramdd via histogramdd.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.histogramdd(*args, **kwargs)


@numpy_eager_registry.register("Iscomplex")
def _np_iscomplex_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Iscomplex via iscomplex.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.iscomplex(*args, **kwargs)


@numpy_eager_registry.register("Iscomplexobj")
def _np_iscomplexobj_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Iscomplexobj via iscomplexobj.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.iscomplexobj(*args, **kwargs)


@numpy_eager_registry.register("Isdtype")
def _np_issubdtype_op_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isdtype via issubdtype.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.issubdtype(*args, **kwargs)


@numpy_eager_registry.register("Isreal")
def _np_isreal_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isreal via isreal.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.isreal(*args, **kwargs)


@numpy_eager_registry.register("Isrealobj")
def _np_isrealobj_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isrealobj via isrealobj.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.isrealobj(*args, **kwargs)


@numpy_eager_registry.register("Isscalar")
def _np_isscalar_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isscalar via isscalar.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.isscalar(*args, **kwargs)


@numpy_eager_registry.register("Issubdtype")
def _np_issubdtype_issubdtype_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Issubdtype via issubdtype.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.issubdtype(*args, **kwargs)


@numpy_eager_registry.register("ResultType")
def _np_result_type_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ResultType via result_type.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.result_type(*args, **kwargs)


@numpy_eager_registry.register("AssertOp")
def _np_assertop(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement AssertOp.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    condition = args[0] if len(args) > 0 else kwargs.get("condition", None)
    if condition is not None:
        assert np.all(np.asarray(condition))
    return np.array([0.0])
