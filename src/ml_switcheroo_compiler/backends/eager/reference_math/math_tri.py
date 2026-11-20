"""Generic backend-parametric eager reference operations for `reference_math.math_tri`."""

from __future__ import annotations

import types
import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _get_np_arg
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.linalg.linear_operator import (
        LinearOperatorBlockLowerTriangular,
        LinearOperatorFullMatrix,
        LinearOperatorLowerTriangular,
        LinearOperatorTridiag,
    )
    from ml_switcheroo_compiler.ops.text.api import AsStringConfig


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        BackendArg: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("Tri")
def _np_tri(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | str | None,
) -> EagerTensorProtocol:
    """Construct an array with ones at and below the given diagonal and zeros elsewhere.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (N, M, k, dtype).
        **kwargs (int | str | None): Keyword arguments.

    Returns:
        BackendArg: Triangular array with ones at and below diagonal.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return _ensure_arr(adapt_backend_module(arr_mod).require_symbol("tri")(*args, **kwargs))


@global_eager_registry.register("TrilIndices")
def _np_trilindices(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | None,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Return the indices for the lower-triangle of an (n, m) array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (n, k, m).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: Tuple of row and column index arrays.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    res = adapt_backend_module(arr_mod).require_symbol("tril_indices")(*args, **kwargs)
    return (_ensure_arr(res[0]), _ensure_arr(res[1]))


@global_eager_registry.register("TrilIndicesFrom")
def _np_trilindicesfrom(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | None,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Return the indices for the lower-triangle of an array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (arr, k).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: Indices for lower triangle of input array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    res = adapt_backend_module(arr_mod).require_symbol("tril_indices_from")(
        arr_mod.asarray(args[0]), *args[1:], **kwargs
    )
    return (_ensure_arr(res[0]), _ensure_arr(res[1]))


@global_eager_registry.register("TrimZeros")
def _np_trimzeros(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | None,
) -> EagerTensorProtocol:
    """Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (filt, trim).
        **kwargs (Union[str, None]): Keyword arguments.

    Returns:
        BackendArg: Trimmed array without leading/trailing zeros.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("trim_zeros")(arr_mod.asarray(args[0]), *args[1:], **kwargs)
    )


@global_eager_registry.register("TriuIndices")
def _np_triuindices(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | None,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Return the indices for the upper-triangle of an (n, m) array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (n, k, m).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: Tuple of row and column index arrays.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    res = adapt_backend_module(arr_mod).require_symbol("triu_indices")(*args, **kwargs)
    return (_ensure_arr(res[0]), _ensure_arr(res[1]))


@global_eager_registry.register("TriuIndicesFrom")
def _np_triuindicesfrom(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | None,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Return the indices for the upper-triangle of an array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (arr, k).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: Indices for upper triangle of input array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    res = adapt_backend_module(arr_mod).require_symbol("triu_indices_from")(
        arr_mod.asarray(args[0]), *args[1:], **kwargs
    )
    return (_ensure_arr(res[0]), _ensure_arr(res[1]))


@global_eager_registry.register("Fromstring")
def _np_fromstring_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: str | str | int | None,
) -> EagerTensorProtocol:
    """A new 1-D array initialized from text data in a string.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (string, dtype, count, sep).
        **kwargs (str | str | int | None): Keyword arguments.

    Returns:
        BackendArg: 1-D array created from string.
    """
    return _ensure_arr(
        adapt_backend_module(backend_module).require_symbol(
            "frombuffer"
            if "sep" not in kwargs and adapt_backend_module(backend_module).has_symbol("frombuffer")
            else "fromstring"
        )(*args, **kwargs)
    )


@global_eager_registry.register("MatrixPower")
def _np_linalg_matrix_power_(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | None,
) -> EagerTensorProtocol:
    """Raise a square matrix to the (integer) power n.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (a, n).
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        BackendArg: Matrix raised to integer power n.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("linalg").matrix_power(*args, **kwargs))


@global_eager_registry.register("AsStringConfig")
def _np_asstringconfig(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | bool | str | None,
) -> AsStringConfig:
    """Construct an AsStringConfig instance.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments for AsStringConfig.
        **kwargs (Union[int, bool, str, None]): Keyword arguments.

    Returns:
        AsStringConfig: Config instance for string formatting.
    """
    from ml_switcheroo_compiler.ops.text.api import AsStringConfig

    return AsStringConfig(*args, **kwargs)


@global_eager_registry.register("LinearOperatorBlockLowerTriangular")
def _np_linearoperatorblocklowertriangular(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | str | None,
) -> LinearOperatorBlockLowerTriangular:
    """Construct a LinearOperatorBlockLowerTriangular instance.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        LinearOperatorBlockLowerTriangular: Block lower triangular operator instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorBlockLowerTriangular

    return LinearOperatorBlockLowerTriangular(*args, **kwargs)


@global_eager_registry.register("LinearOperatorFullMatrix")
def _np_linearoperatorfullmatrix(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | str | None,
) -> LinearOperatorFullMatrix:
    """Construct a LinearOperatorFullMatrix instance.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        LinearOperatorFullMatrix: Full matrix linear operator instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorFullMatrix

    return LinearOperatorFullMatrix(*args, **kwargs)


@global_eager_registry.register("LinearOperatorLowerTriangular")
def _np_linearoperatorlowertriangular(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | str | None,
) -> LinearOperatorLowerTriangular:
    """Construct a LinearOperatorLowerTriangular instance.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        LinearOperatorLowerTriangular: Lower triangular operator instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorLowerTriangular

    return LinearOperatorLowerTriangular(*args, **kwargs)


@global_eager_registry.register("LinearOperatorTridiag")
def _np_linearoperatortridiag(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | str | None,
) -> LinearOperatorTridiag:
    """Construct a LinearOperatorTridiag instance.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments.
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        LinearOperatorTridiag: Tridiagonal linear operator instance.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorTridiag

    return LinearOperatorTridiag(*args, **kwargs)


@global_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | None,
) -> EagerTensorProtocol | None:
    """Evaluate confusion matrix of true and predicted labels.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (y_true, y_pred).
        **kwargs (Union[int, None]): Keyword arguments (e.g. num_classes).

    Returns:
        Union[EagerTensorProtocol, None]: 2D confusion matrix or None if labels missing.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    y_true = _get_np_arg(args, 0)
    y_pred = _get_np_arg(args, 1)
    if y_true is None or y_pred is None:
        return None
    num_classes = kwargs.get("num_classes", None)
    if num_classes is None:
        num_classes = max(int(coerce_int(arr_mod.max(y_true).item())), int(coerce_int(arr_mod.max(y_pred).item()))) + 1
    return _ensure_arr(
        arr_mod.bincount(arr_mod.add(arr_mod.multiply(y_true, num_classes), y_pred), minlength=num_classes**2).reshape(
            (num_classes, num_classes)
        )
    )


@global_eager_registry.register("Distributions")
def _np_distributions(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | str | None,
) -> dict[str, EagerTensorProtocol] | None:
    """Compute histogram counts and bin edges for distribution inspection.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments (a).
        **kwargs (Union[int, str, None]): Keyword arguments.

    Returns:
        Union[dict[str, EagerTensorProtocol], None]: Dictionary with counts and bins or None if input missing.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    counts, bins = arr_mod.histogram(a, bins="auto")
    return {"counts": _ensure_arr(counts), "bins": _ensure_arr(bins)}


@global_eager_registry.register("StridedSlice")
def _np_stridedslice(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    data: BackendArg,
    start: Sequence[int],
    end: Sequence[int],
    strides: Sequence[int],
    **kwargs: int | None,
) -> EagerTensorProtocol:
    """Evaluate strided slice eagerly on an array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        data (BackendArg): Input array to slice.
        start (Sequence[int]): Starting indices per dimension.
        end (Sequence[int]): Ending indices per dimension.
        strides (Sequence[int]): Step increments per dimension.
        **kwargs (Union[int, None]): Keyword arguments.

    Returns:
        BackendArg: Sliced view or copy of data array.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    slices = tuple(slice(s, e, st) for s, e, st in zip(start, end, strides))
    return coerce_array(backend_module, data)[slices]
