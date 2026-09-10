"""Dedicated eager execution handlers and dispatch for the Numba backend."""

import numpy as np

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def execute_op(
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
    schema = mapping_loader.load_backend_mappings("numba")
    if op_type in schema.operations:
        processed_args: list[object] = []
        for arg in args:
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

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(np, *args, **kwargs)

    msg = f"Operation '{op_type}' is not implemented for Numba backend."
    raise BackendNotSupportedError(msg)
