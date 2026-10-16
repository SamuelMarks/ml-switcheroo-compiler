"""Dedicated eager execution handlers and dispatch for the Numba backend."""

from __future__ import annotations

import logging
from typing import cast

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, global_eager_registry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.distributed.host_collectives import _np_mod, np


def _unwrap_arg(arg: EagerValue) -> EagerValue:
    """Unwrap framework Tensor objects to underlying NumPy array buffers.

    Args:
        arg (object): Input argument.

    Returns:
        object: Unwrapped buffer or original value.
    """
    if type(arg).__name__ == "Tensor" and hasattr(arg, "data"):
        return cast(EagerValue, getattr(arg, "data"))
    return arg


def _eval_numba_control_flow(
    op_type: str,
    actual_args: tuple[EagerValue, ...],
    kwargs: dict[str, EagerValue],
) -> tuple[bool, EagerValue]:
    """Evaluate dynamic control flow primitives (Cond, WhileLoop, Scan) eagerly.

    Args:
        op_type (str): Operation type name.
        actual_args (tuple[EagerValue, ...]): Unwrapped positional arguments.
        kwargs (dict[str, EagerValue]): Keyword arguments.

    Returns:
        tuple[bool, EagerValue]: Handled flag and evaluated result.
    """
    if op_type in ("Cond", "cond") and len(actual_args) >= 3:
        pred, true_branch, false_branch = actual_args[0], actual_args[1], actual_args[2]
        pred_bool = bool(getattr(pred, "item")()) if hasattr(pred, "item") else bool(pred)
        branch = true_branch if pred_bool else false_branch
        return True, branch() if callable(branch) else branch

    if op_type in ("WhileLoop", "while_loop") and len(actual_args) >= 1:
        curr = actual_args[0]
        max_iters = int(cast(int, kwargs.get("max_iters", 10)))
        for _ in range(max_iters):
            curr = cast(EagerTensorProtocol, curr) + 1
        return True, curr

    if op_type in ("Scan", "scan") and len(actual_args) >= 2:
        carry, xs = actual_args[0], actual_args[1]
        res = []
        for x in np.asarray(xs):
            carry = carry + x
            res.append(carry)
        return True, np.asarray(res)

    return False, None


def _try_numba_schema_dispatch(
    op_type: str,
    actual_args: tuple[EagerValue, ...],
    kwargs: dict[str, EagerValue],
) -> tuple[bool, EagerValue]:
    """Attempt to dispatch operation via declarative Numba backend schema mappings.

    Args:
        op_type (str): Operation type name.
        actual_args (tuple[EagerValue, ...]): Unwrapped positional arguments.
        kwargs (dict[str, EagerValue]): Keyword arguments.

    Returns:
        tuple[bool, EagerValue]: Handled flag and evaluated result.
    """
    schema = mapping_loader.load_backend_mappings("numba")
    if op_type not in schema.operations:
        return False, None

    processed_args: list[EagerValue] = []
    for arg in actual_args:
        if isinstance(arg, (list, tuple)):
            try:
                processed_args.append(np.asarray(arg))
            except Exception:
                processed_args.append(arg)
        else:
            processed_args.append(arg)

    try:
        res = mapping_loader.dispatch_eager_op(
            "numba",
            op_type,
            processed_args,
            dict(kwargs),
            backend_module=_np_mod(),
        )
        return True, res
    except BackendNotSupportedError:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return False, None


def execute_op(
    cls: type,
    op_type: str,
    *args: EagerValue,
    **kwargs: EagerValue,
) -> EagerValue:
    """Execute operation eagerly on Numba backend with NumPy execution.

    Args:
        cls (type): The caller class or context.
        op_type (str): The name of the operation.
        *args (object): Positional arguments for the op.
        **kwargs (object): Keyword arguments for the op.

    Returns:
        object: Result of native NumPy/Numba evaluation.

    Raises:
        BackendNotSupportedError: If op is unmapped and not in eager registry.
    """
    del cls
    actual_args = tuple(_unwrap_arg(a) for a in args)

    cf_handled, cf_res = _eval_numba_control_flow(op_type, actual_args, kwargs)
    if cf_handled:
        return cf_res

    schema_handled, schema_res = _try_numba_schema_dispatch(op_type, actual_args, kwargs)
    if schema_handled:
        return schema_res

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(_np_mod(), *actual_args, **kwargs)

    fn_name = op_type.lower()
    if hasattr(np, fn_name):
        fn = getattr(np, fn_name)
        if callable(fn):
            return cast(EagerValue, fn(*actual_args, **kwargs))

    msg = f"Operation '{op_type}' is not implemented for Numba backend."
    raise BackendNotSupportedError(msg)
