# ruff: noqa: E501
"""Backend utilities."""

try:
    import cupy as cp
except ImportError:
    cp = None


_OP_MAPPING = None


def _get_op_mapping() -> dict:
    """Retrieve the operation mapping for the CuPy backend.

    Returns:
        dict: A dictionary mapping operation types to their implementations.
    """
    global _OP_MAPPING
    if _OP_MAPPING is not None:
        return _OP_MAPPING
    _OP_MAPPING = {}
    return _OP_MAPPING


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute an eager operation using the CuPy backend.

    Args:
        cls (type): The tensor class.
        op_type (str): The name of the operation to execute.
        *args (object): Positional arguments for the operation.
        **kwargs (object): Keyword arguments for the operation.

    Returns:
        object: The result of the operation execution.

    Raises:
        BackendNotSupportedError: If the operation is not supported by the CuPy backend.
    """
    import ml_switcheroo_compiler.backends.eager  # noqa: F401
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(cls, *args, **kwargs)

    op_mapping = _get_op_mapping()
    func = op_mapping.get(op_type)
    if func is not None:
        return func(*args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.") from None
