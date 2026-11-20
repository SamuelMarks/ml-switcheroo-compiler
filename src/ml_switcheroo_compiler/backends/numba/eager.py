"""Dedicated eager execution handlers and dispatch for the Numba backend."""

from __future__ import annotations

import logging
import typing

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_operand_list,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.numba import kernels
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.shape import DataHolderProtocol


@typing.runtime_checkable
class _ItemCallableProtocol(typing.Protocol):
    """Protocol for scalar tensors exposing `.item()`."""

    def item(self) -> BackendArg:
        """Extract scalar item.

        Returns:
            BackendArg: Scalar item.
        """
        ...


def _unwrap_arg(arg: BackendArg) -> BackendArg:
    """Unwrap framework Tensor objects to underlying array buffers.

    Args:
        arg (BackendArg): Input argument.

    Returns:
        BackendArg: Unwrapped buffer or original value.
    """
    raw_arg: BackendArg = arg
    if type(arg).__name__ == "Tensor" and isinstance(raw_arg, DataHolderProtocol):
        return raw_arg.data
    return arg


def _eval_numba_control_flow(
    op_type: str,
    actual_args: tuple[BackendArg, ...],
    kwargs: dict[str, BackendArg],
) -> tuple[bool, BackendArg]:
    """Evaluate dynamic control flow primitives (Cond, WhileLoop, Scan) eagerly.

    Args:
        op_type (str): Operation type name.
        actual_args (tuple[BackendArg, ...]): Unwrapped positional arguments.
        kwargs (dict[str, BackendArg]): Keyword arguments.

    Returns:
        tuple[bool, BackendArg]: Handled flag and evaluated result.
    """
    if op_type in ("Cond", "cond") and len(actual_args) >= 3:
        pred, true_branch, false_branch = actual_args[0], actual_args[1], actual_args[2]
        raw_pred: BackendArg | _ItemCallableProtocol = pred
        pred_bool = bool(raw_pred.item()) if isinstance(raw_pred, _ItemCallableProtocol) else bool(pred)
        branch = true_branch if pred_bool else false_branch
        return True, branch() if callable(branch) else branch

    if op_type in ("WhileLoop", "while_loop") and len(actual_args) >= 1:
        from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

        curr: BackendArg = actual_args[0]
        max_iters = int(coerce_int(kwargs.get("max_iters", 10)))
        for _ in range(max_iters):
            curr = coerce_numeric(curr) + 1
        return True, curr

    if op_type in ("Scan", "scan") and len(actual_args) >= 2:
        from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

        carry: BackendArg = actual_args[0]
        xs = actual_args[1]
        res = []
        xs_arr = kernels.asarray(xs)
        xs_list = coerce_operand_list(adapt_backend_module(xs_arr).require_op("tolist")())
        for x in xs_list:
            carry = coerce_numeric(carry) + coerce_numeric(x)
            res.append(carry)
        return True, kernels.asarray(res)

    return False, None


def _try_numba_schema_dispatch(
    op_type: str,
    actual_args: tuple[BackendArg, ...],
    kwargs: dict[str, BackendArg],
) -> tuple[bool, BackendArg]:
    """Attempt to dispatch operation via declarative Numba backend schema mappings.

    Args:
        op_type (str): Operation type name.
        actual_args (tuple[BackendArg, ...]): Unwrapped positional arguments.
        kwargs (dict[str, BackendArg]): Keyword arguments.

    Returns:
        tuple[bool, BackendArg]: Handled flag and evaluated result.
    """
    schema = mapping_loader.load_backend_mappings("numba")
    if op_type not in schema.operations:
        return False, None

    processed_args: list[BackendArg] = []
    for arg in actual_args:
        if isinstance(arg, (list, tuple)):
            try:
                processed_args.append(kernels.asarray(arg))
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
                processed_args.append(arg)
        else:
            processed_args.append(arg)

    try:
        res = mapping_loader.dispatch_eager_op(
            "numba",
            op_type,
            processed_args,
            dict(kwargs),
            backend_module=kernels,
        )
        return True, res
    except BackendNotSupportedError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return False, None


def execute_op(
    op_type: str,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute operation eagerly on Numba backend.

    Args:
        op_type (str): The name of the operation.
        *args (BackendArg): Positional arguments for the op.
        **kwargs (BackendArg): Keyword arguments for the op.

    Returns:
        BackendArg: Result of Numba evaluation.

    Raises:
        BackendNotSupportedError: If op is unmapped and not in eager registry.
    """
    actual_args = tuple(_unwrap_arg(a) for a in args)

    cf_handled, cf_res = _eval_numba_control_flow(op_type, actual_args, kwargs)
    if cf_handled:
        return cf_res

    schema_handled, schema_res = _try_numba_schema_dispatch(op_type, actual_args, kwargs)
    if schema_handled:
        return schema_res

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(kernels, *actual_args, **kwargs)

    fn_name = op_type.lower()
    has_fn, fn = adapt_backend_module(kernels).resolve_symbol(fn_name)
    if has_fn and callable(fn):
        return fn(*actual_args, **kwargs)

    msg = f"Operation '{op_type}' is not implemented for Numba backend."
    raise BackendNotSupportedError(msg)
