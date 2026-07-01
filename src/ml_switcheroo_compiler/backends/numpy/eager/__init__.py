"""Numpy Eager Evaluator package."""

import importlib
import pkgutil
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
import numpy as np
import re


# Auto-discover all modules in this package to run their @register decorators

for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute execute_op.

    Args:
        cls (Any): The class.
        op_type (Any): Argument op_type.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    func_registry = numpy_eager_registry.get(op_type)
    if func_registry is not None:
        return func_registry(np, *args, **kwargs)

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func_registry = global_eager_registry.get(op_type)
    if func_registry is not None:
        return func_registry(np, *args, **kwargs)

    try:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        func = getattr(np, snake)
    except AttributeError:
        msg = f"Operation {op_type} is not implemented in interpreter."
        raise NotImplementedError(msg) from None

    return func(*args, **kwargs)


__all__ = [
    "execute_op",
]
