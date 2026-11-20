"""Generic backend-parametric eager reference operations for `reference_math.math_matrix_utils`."""

from __future__ import annotations

import logging
import types
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _get_np_arg
from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("DiagIndices")
def _np_diag_indices_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement DiagIndices via diag_indices.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("diag_indices")(*args, **kwargs)


@global_eager_registry.register("DiagIndicesFrom")
def _np_diag_indices_from_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement DiagIndicesFrom via diag_indices_from.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("diag_indices_from")(*args, **kwargs)


@global_eager_registry.register("Diagflat")
def _np_diagflat_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Diagflat via diagflat.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("diagflat")(*args, **kwargs)


@global_eager_registry.register("Diagonal")
def _np_diagonal_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Diagonal via diagonal.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("diagonal")(*args, **kwargs)


@global_eager_registry.register("Indices")
def _np_indices_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Indices via indices.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("indices")(*args, **kwargs)


@global_eager_registry.register("MaskIndices")
def _np_mask_indices_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement MaskIndices via mask_indices.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_op("mask_indices")(*args, **kwargs)


@global_eager_registry.register("LinearOperatorBlockDiag")
def _np_linearoperatorblockdiag(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement LinearOperatorBlockDiag.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorBlockDiag

    return LinearOperatorBlockDiag(*args, **kwargs)


@global_eager_registry.register("LinearOperatorDiag")
def _np_linearoperatordiag(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement LinearOperatorDiag.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorDiag

    return LinearOperatorDiag(*args, **kwargs)


@global_eager_registry.register("confusion_matrix")
def _np_confusion_matrix(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np confusion matrix operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    from ml_switcheroo_compiler.backends.eager_registry import require_callable

    arr_mod = _resolve_eager_mod(backend_module)
    try:
        import ml_switcheroo_compiler.ops as _ops

        if adapt_backend_module(_ops).has_symbol("confusion_matrix"):
            cls_or_func: BackendArg = adapt_backend_module(_ops).require_symbol("confusion_matrix")
            op_def_cls: BackendArg = adapt_backend_module(_ops).get_symbol("OpDef", type)
            if (
                isinstance(cls_or_func, type)
                and isinstance(op_def_cls, type)
                and not issubclass(cls_or_func, op_def_cls)
            ):
                return require_callable(cls_or_func)(*args, **kwargs)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ):
        # Ignore dynamic load errors
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    if adapt_backend_module(backend_module).has_symbol("confusion_matrix"):
        return adapt_backend_module(backend_module).require_symbol("confusion_matrix")(*args, **kwargs)
    y_true = arr_mod.asarray(args[0]).flatten()
    y_pred = arr_mod.asarray(args[1]).flatten()

    raw_num_classes = (
        kwargs.get("num_classes", args[2] if len(args) > 2 else None)
        if adapt_backend_module(kwargs).has_symbol("get")
        else None
    )
    num_classes: int = (
        int(arr_mod.max(y_true) if arr_mod.max(y_true) > arr_mod.max(y_pred) else arr_mod.max(y_pred)) + 1
        if raw_num_classes is None
        else int(coerce_int(raw_num_classes))
    )
    return arr_mod.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)


@global_eager_registry.register("distributions")
def _np_distributions(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np distributions operation on the active backend array.

    Args:
        backend_module: The backend.
        args: Positional args.
        kwargs: Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import require_callable

    arr_mod = _resolve_eager_mod(backend_module)
    try:
        import ml_switcheroo_compiler.ops as _ops

        if adapt_backend_module(_ops).has_symbol("distributions"):
            cls_or_func: BackendArg = adapt_backend_module(_ops).require_symbol("distributions")
            op_def_cls: BackendArg = adapt_backend_module(_ops).get_symbol("OpDef", type)
            if (
                isinstance(cls_or_func, type)
                and isinstance(op_def_cls, type)
                and not issubclass(cls_or_func, op_def_cls)
            ):
                return require_callable(cls_or_func)(*args, **kwargs)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        print("EXCEPTION IN _np_distributions:", repr(e))
        # Ignore dynamic load errors
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    if adapt_backend_module(backend_module).has_symbol("distributions"):
        return adapt_backend_module(backend_module).require_symbol("distributions")(*args, **kwargs)
    arr = arr_mod.asarray(args[0]) if args else arr_mod.zeros((1,))
    return arr_mod.array([arr_mod.mean(arr), arr_mod.var(arr)])


@global_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix_cap(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np confusion matrix cap operation on the active backend array.

    Args:
        backend_module: The backend.
        args: Positional args.
        kwargs: Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = _get_np_arg(args, 0)
    b = _get_np_arg(args, 1)
    if a is None or b is None:
        return None
    return arr_mod.zeros((2, 2))
