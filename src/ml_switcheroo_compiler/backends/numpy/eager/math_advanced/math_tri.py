# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Math Ops."""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_misc_ext import _get_np_arg, _get_sc


@numpy_eager_registry.register("Tri")
def _np_tri(backend_module: object, *args: object, **kwargs: object) -> object:
    """Construct an array with ones at and below the given diagonal and zeros elsewhere.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return np.tri(*args, **kwargs)


@numpy_eager_registry.register("TrilIndices")
def _np_trilindices(backend_module: object, *args: object, **kwargs: object) -> object:
    """Return the indices for the lower-triangle of an (n, m) array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return np.tril_indices(*args, **kwargs)


@numpy_eager_registry.register("TrilIndicesFrom")
def _np_trilindicesfrom(backend_module: object, *args: object, **kwargs: object) -> object:
    """Return the indices for the lower-triangle of an array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return np.tril_indices_from(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("TrimZeros")
def _np_trimzeros(backend_module: object, *args: object, **kwargs: object) -> object:
    """Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return np.trim_zeros(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("TriuIndices")
def _np_triuindices(backend_module: object, *args: object, **kwargs: object) -> object:
    """Return the indices for the upper-triangle of an (n, m) array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return np.triu_indices(*args, **kwargs)


@numpy_eager_registry.register("TriuIndicesFrom")
def _np_triuindicesfrom(backend_module: object, *args: object, **kwargs: object) -> object:
    """Return the indices for the upper-triangle of an array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return np.triu_indices_from(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Fromstring")
def _np_fromstring_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement Fromstring via fromstring.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.fromstring(*args, **kwargs)


@numpy_eager_registry.register("MatrixPower")
def _np_linalg_matrix_power_(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement MatrixPower via linalg.matrix_power.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: object: The computed result.
    """
    return backend_module.linalg.matrix_power(*args, **kwargs)


@numpy_eager_registry.register("AsStringConfig")
def _np_asstringconfig(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement AsStringConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.text.frontend import AsStringConfig

    return AsStringConfig(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorBlockLowerTriangular")
def _np_linearoperatorblocklowertriangular(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement LinearOperatorBlockLowerTriangular.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorBlockLowerTriangular

    return LinearOperatorBlockLowerTriangular(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorFullMatrix")
def _np_linearoperatorfullmatrix(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement LinearOperatorFullMatrix.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorFullMatrix

    return LinearOperatorFullMatrix(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorLowerTriangular")
def _np_linearoperatorlowertriangular(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement LinearOperatorLowerTriangular.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorLowerTriangular

    return LinearOperatorLowerTriangular(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorTridiag")
def _np_linearoperatortridiag(backend_module: object, *args: object, **kwargs: object) -> object:
    """Implement LinearOperatorTridiag.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorTridiag

    return LinearOperatorTridiag(*args, **kwargs)


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_confusion_matrix operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    y_true: object = _get_np_arg(args, 0)
    y_pred: object = _get_np_arg(args, 1)
    if y_true is None or y_pred is None:
        return None
    num_classes: object = kwargs.get("num_classes", None)
    if num_classes is None:
        num_classes: object = max(np.max(y_true), np.max(y_pred)) + 1
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape((num_classes, num_classes))


@numpy_eager_registry.register("Distributions")
def _np_distributions(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_distributions operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a: object = _get_np_arg(args, 0)
    if a is None:
        return None
    counts, bins = np.histogram(a, bins="auto")
    return {"counts": counts, "bins": bins}


@numpy_eager_registry.register("StridedSlice")
def _np_stridedslice(backend_module: object, data: object, start: object, end: object, strides: object, **kwargs: object) -> object:  # noqa: D417
    """Evaluate _np_stridedslice logic eagerly backed by NumPy.

    Args:
        backend_module (object): The backend_module parameter.
        data (object): The data parameter.
        start (object): The start parameter.
        end (object): The end parameter.
        strides (object): The strides parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    slices: object = tuple(slice(s, e, st) for s, e, st in zip(start, end, strides))
    return data[slices]
