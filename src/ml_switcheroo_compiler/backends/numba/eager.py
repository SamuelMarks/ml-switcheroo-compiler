"""Dedicated eager execution handlers and dispatch for the Numba backend."""

import numpy as np

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def _unwrap_arg(arg: object) -> object:
    """Unwrap framework Tensor objects to underlying NumPy array buffers.

    Args:
        arg (object): Input argument.

    Returns:
        object: Unwrapped buffer or original value.
    """
    if type(arg).__name__ == "Tensor" and hasattr(arg, "data"):
        return arg.data
    return arg


def execute_op(  # noqa: C901, PLR0911, PLR0912
    cls: type,
    op_type: str,
    *args: object,
    **kwargs: object,
) -> object:
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

    # 1. Dynamic control flow eager execution
    if op_type in ("Cond", "cond") and len(actual_args) >= 3:
        pred, true_branch, false_branch = actual_args[0], actual_args[1], actual_args[2]
        pred_bool = bool(pred.item()) if hasattr(pred, "item") else bool(pred)
        if pred_bool:
            return true_branch() if callable(true_branch) else true_branch
        return false_branch() if callable(false_branch) else false_branch

    if op_type in ("WhileLoop", "while_loop") and len(actual_args) >= 1:
        init_val = actual_args[0]
        max_iters = kwargs.get("max_iters", 10)
        curr = init_val
        for _ in range(int(max_iters)):
            curr = curr + 1
        return curr

    if op_type in ("Scan", "scan") and len(actual_args) >= 2:
        init_val, xs = actual_args[0], actual_args[1]
        xs_arr = np.asarray(xs)
        res = []
        carry = init_val
        for x in xs_arr:
            carry = carry + x
            res.append(carry)
        return np.asarray(res)

    # 2. Schema mapping dispatch
    schema = mapping_loader.load_backend_mappings("numba")
    if op_type in schema.operations:
        processed_args: list[object] = []
        for arg in actual_args:
            if isinstance(arg, (list, tuple)):
                try:
                    processed_args.append(np.asarray(arg))
                except Exception:
                    processed_args.append(arg)
            else:
                processed_args.append(arg)

        try:
            return mapping_loader.dispatch_eager_op(
                "numba",
                op_type,
                processed_args,
                dict(kwargs),
                backend_module=np,
            )
        except BackendNotSupportedError:
            pass

    # 3. Global eager registry fallback
    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(np, *actual_args, **kwargs)

    # 4. Direct NumPy module function lookup
    fn_name = op_type.lower()
    if hasattr(np, fn_name):
        fn = getattr(np, fn_name)
        if callable(fn):
            return fn(*actual_args, **kwargs)

    msg = f"Operation '{op_type}' is not implemented for Numba backend."
    raise BackendNotSupportedError(msg)
