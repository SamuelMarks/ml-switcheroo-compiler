"""Dedicated eager execution handlers and dispatch for the TensorFlow backend."""

import sys

from ml_switcheroo_compiler.backends.mapping_loader import dispatch_eager_op, load_backend_mappings
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def execute_op(
    cls: type,
    op_type: str,
    *args: object,
    **kwargs: object,
) -> object:
    """Execute operation eagerly on TensorFlow backend with native argument validation.

    Args:
        cls (type): The caller class or context.
        op_type (str): The name of the operation.
        *args (object): Positional arguments for the op.
        **kwargs (object): Keyword arguments for the op.

    Returns:
        object: Result of native TensorFlow evaluation.

    Raises:
        BackendNotSupportedError: If TensorFlow is unavailable or op is unmapped.
    """
    tf = sys.modules.get("tensorflow")
    if tf is None:
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            tf = None

    if tf is None:
        raise BackendNotSupportedError("TensorFlow is not installed or available in this environment.")

    # Convert Python sequences to native TensorFlow tensor primitives where possible
    processed_args: list[object] = []
    for arg in args:
        if isinstance(arg, (list, tuple)) and hasattr(tf, "convert_to_tensor"):
            try:
                processed_args.append(tf.convert_to_tensor(arg))
            except Exception:
                processed_args.append(arg)
        else:
            processed_args.append(arg)

    schema = load_backend_mappings("tensorflow")
    if op_type in schema.operations:
        return dispatch_eager_op("tensorflow", op_type, processed_args, dict(kwargs), backend_module=tf)

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(tf, *args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not supported by tensorflow eager backend.")
