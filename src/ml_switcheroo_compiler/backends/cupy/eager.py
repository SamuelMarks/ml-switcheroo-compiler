# ruff: noqa: E501
"""Backend utilities."""

try:
    import cupy as cp
except ImportError:
    cp = None


_OP_MAPPING = None


def _get_op_mapping() -> dict:
    global _OP_MAPPING
    if _OP_MAPPING is not None:
        return _OP_MAPPING
    _OP_MAPPING = {}
    return _OP_MAPPING


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
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(cls, *args, **kwargs)

    op_mapping = _get_op_mapping()
    func = op_mapping.get(op_type)
    if func is not None:
        return func(*args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.") from None
