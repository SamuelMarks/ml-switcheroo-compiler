"""Triangular and matrix manipulation operations for numpy eager backend."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerValue,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general import _get_np_arg
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.linalg.linear_operator import (
        LinearOperatorBlockLowerTriangular,
        LinearOperatorFullMatrix,
        LinearOperatorLowerTriangular,
        LinearOperatorTridiag,
    )
    from ml_switcheroo_compiler.ops.text.frontend import AsStringConfig


@numpy_eager_registry.register("Tri")
def _np_tri(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | np.dtype[np.float64] | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Construct an array with ones at and below the given diagonal and zeros elsewhere.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, np.dtype]): Positional arguments (N, M, k, dtype).
        **kwargs (Union[int, np.dtype, None]): Keyword arguments.

    Returns:
        np.ndarray: Triangular array with ones at and below diagonal.
    """
    return as_ndarray(getattr(np, "tri")(*args, **kwargs))


@numpy_eager_registry.register("TrilIndices")
def _np_trilindices(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | None,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Return the indices for the lower-triangle of an (n, m) array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, None]): Positional arguments (n, k, m).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of row and column index arrays.
    """
    res = getattr(np, "tril_indices")(*args, **kwargs)
    return (as_ndarray(res[0]), as_ndarray(res[1]))


@numpy_eager_registry.register("TrilIndicesFrom")
def _np_trilindicesfrom(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | None,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Return the indices for the lower-triangle of an array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], int]): Positional arguments (arr, k).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        tuple[np.ndarray, np.ndarray]: Indices for lower triangle of input array.
    """
    res = getattr(np, "tril_indices_from")(np.asarray(args[0]), *args[1:], **kwargs)
    return (as_ndarray(res[0]), as_ndarray(res[1]))


@numpy_eager_registry.register("TrimZeros")
def _np_trimzeros(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], list[float], str]): Positional arguments (filt, trim).
        **kwargs (Union[str, None]): Keyword arguments.

    Returns:
        np.ndarray: Trimmed array without leading/trailing zeros.
    """
    return as_ndarray(getattr(np, "trim_zeros")(np.asarray(args[0]), *args[1:], **kwargs))


@numpy_eager_registry.register("TriuIndices")
def _np_triuindices(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | None,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Return the indices for the upper-triangle of an (n, m) array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, None]): Positional arguments (n, k, m).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of row and column index arrays.
    """
    res = getattr(np, "triu_indices")(*args, **kwargs)
    return (as_ndarray(res[0]), as_ndarray(res[1]))


@numpy_eager_registry.register("TriuIndicesFrom")
def _np_triuindicesfrom(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | None,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Return the indices for the upper-triangle of an array.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], int]): Positional arguments (arr, k).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        tuple[np.ndarray, np.ndarray]: Indices for upper triangle of input array.
    """
    res = getattr(np, "triu_indices_from")(np.asarray(args[0]), *args[1:], **kwargs)
    return (as_ndarray(res[0]), as_ndarray(res[1]))


@numpy_eager_registry.register("Fromstring")
def _np_fromstring_(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: np.dtype[np.float64] | str | int | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """A new 1-D array initialized from text data in a string.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[str, bytes, np.dtype, int]): Positional arguments (string, dtype, count, sep).
        **kwargs (Union[np.dtype, str, int, None]): Keyword arguments.

    Returns:
        np.ndarray: 1-D array created from string.
    """
    return as_ndarray(getattr(backend_module, "frombuffer" if "sep" not in kwargs and hasattr(backend_module, "frombuffer") else "fromstring")(*args, **kwargs))


@numpy_eager_registry.register("MatrixPower")
def _np_linalg_matrix_power_(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Raise a square matrix to the (integer) power n.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int]): Positional arguments (a, n).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        np.ndarray: Matrix raised to integer power n.
    """
    return as_ndarray(backend_module.linalg.matrix_power(*args, **kwargs))


@numpy_eager_registry.register("AsStringConfig")
def _np_asstringconfig(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | bool | str | None,
) -> AsStringConfig:
    """Construct an AsStringConfig instance.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[int, bool, str]): Positional arguments for AsStringConfig.
        **kwargs (Union[int, bool, str, None]): Keyword arguments.

    Returns:
        AsStringConfig: Config instance for string formatting.
    """
    from ml_switcheroo_compiler.ops.text.frontend import AsStringConfig

    return AsStringConfig(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorBlockLowerTriangular")
def _np_linearoperatorblocklowertriangular(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | str | None,
) -> LinearOperatorBlockLowerTriangular:
    """Construct a LinearOperatorBlockLowerTriangular instance.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[list[BaseLinearOperator], np.ndarray, int]): Positional arguments.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        LinearOperatorBlockLowerTriangular: Block lower triangular operator instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorBlockLowerTriangular

    return LinearOperatorBlockLowerTriangular(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorFullMatrix")
def _np_linearoperatorfullmatrix(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | str | None,
) -> LinearOperatorFullMatrix:
    """Construct a LinearOperatorFullMatrix instance.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, str]): Positional arguments.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        LinearOperatorFullMatrix: Full matrix linear operator instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorFullMatrix

    return LinearOperatorFullMatrix(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorLowerTriangular")
def _np_linearoperatorlowertriangular(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | str | None,
) -> LinearOperatorLowerTriangular:
    """Construct a LinearOperatorLowerTriangular instance.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, str]): Positional arguments.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        LinearOperatorLowerTriangular: Lower triangular operator instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorLowerTriangular

    return LinearOperatorLowerTriangular(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorTridiag")
def _np_linearoperatortridiag(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | str | None,
) -> LinearOperatorTridiag:
    """Construct a LinearOperatorTridiag instance.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, int, str]): Positional arguments.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        LinearOperatorTridiag: Tridiagonal linear operator instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorTridiag

    return LinearOperatorTridiag(*args, **kwargs)


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None:
    """Evaluate confusion matrix of true and predicted labels.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[int], int]): Positional arguments (y_true, y_pred).
        **kwargs (Union[int, None]): Keyword arguments (e.g. num_classes).

    Returns:
        Union[np.ndarray, None]: 2D confusion matrix or None if labels missing.
    """
    y_true = _get_np_arg(args, 0)
    y_pred = _get_np_arg(args, 1)
    if y_true is None or y_pred is None:
        return None
    num_classes = kwargs.get("num_classes", None)
    if num_classes is None:
        num_classes = max(int(np.max(y_true).item()), int(np.max(y_pred).item())) + 1
    return as_ndarray(np.bincount(np.add(np.multiply(y_true, num_classes), y_pred), minlength=num_classes**2).reshape((num_classes, num_classes)))


@numpy_eager_registry.register("Distributions")
def _np_distributions(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | str | None,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float64]]] | None:
    """Compute histogram counts and bin edges for distribution inspection.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[float], list[int]]): Positional arguments (a).
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        Union[dict[str, np.ndarray], None]: Dictionary with counts and bins or None if input missing.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    counts, bins = np.histogram(a, bins="auto")
    return {"counts": as_ndarray(counts), "bins": as_ndarray(bins)}


@numpy_eager_registry.register("StridedSlice")
def _np_stridedslice(
    backend_module: BackendModuleProtocol,
    data: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    start: Sequence[int],
    end: Sequence[int],
    strides: Sequence[int],
    **kwargs: int | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Evaluate strided slice eagerly on an array.

    Args:
        backend_module (ModuleType): Active backend module.
        data (np.ndarray): Input array to slice.
        start (Sequence[int]): Starting indices per dimension.
        end (Sequence[int]): Ending indices per dimension.
        strides (Sequence[int]): Step increments per dimension.
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        np.ndarray: Sliced view or copy of data array.
    """
    slices = tuple(slice(s, e, st) for s, e, st in zip(start, end, strides))
    return data[slices]
