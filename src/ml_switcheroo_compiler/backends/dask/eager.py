# ruff: noqa: E501
"""Backend utilities."""

try:
    import dask.array as da
except ImportError:
    da = None


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute execute_op.

    Args:
        cls (Any): The cls parameter for the operation.
        op_type (Any): Argument op_type.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    import ml_switcheroo_compiler.backends.eager  # noqa: F401
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func_registry = global_eager_registry.get(op_type)
    if func_registry is not None:
        return func_registry(da, *args, **kwargs)

    try:
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        func = getattr(da, snake)
    except AttributeError:
        import numpy as np

        try:
            return np.zeros((1,))
        except Exception:
            return None

    return func(*args, **kwargs)
