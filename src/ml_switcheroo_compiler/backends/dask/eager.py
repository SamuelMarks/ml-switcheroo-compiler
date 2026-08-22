# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Backend utilities."""

from typing import Any

try:
    import dask.array as da
except ImportError:
    da = None  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


def execute_op(cls: type, op_type: str, *args: Any, **kwargs: Any) -> Any:
    """Execute an eager operation using the Dask backend.

    Args:
        cls (type): The tensor class.
        op_type (str): The name of the operation to execute.
        *args (object): Positional arguments for the operation.
        **kwargs (object): Keyword arguments for the operation.

    Returns: Any: The result of the operation execution.

    Raises:
        BackendNotSupportedError: If the operation is not supported by the Dask backend.
    """
    import ml_switcheroo_compiler.backends.eager  # noqa: F401
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(cls, *args, **kwargs)
    from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings, resolve_target_api

    schema = load_backend_mappings("dask")
    if op_type in schema.operations and (schema.operations[op_type].target_api or schema.operations[op_type].custom_code):
        import sys

        func = resolve_target_api(schema.operations[op_type].target_api, schema.operations[op_type].custom_code, sys.modules[__name__])
        if func is not None:
            return func(*args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.") from None
