"""Dedicated eager execution handlers and dispatch for the TensorFlow backend."""

from __future__ import annotations

import sys

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.mapping_loader import dispatch_eager_op, load_backend_mappings
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def execute_op(
    cls: type | None,
    op_type: str,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute operation eagerly on TensorFlow backend with native argument validation.

    Args:
        cls (type | None): The caller class or context.
        op_type (str): The name of the operation.
        *args (BackendArg): Positional arguments for the op.
        **kwargs (BackendArg): Keyword arguments for the op.

    Returns:
        BackendArg: Result of native TensorFlow evaluation.

    Raises:
        BackendNotSupportedError: If TensorFlow is unavailable or op is unmapped.
    """
    tf = sys.modules.get("tensorflow")
    if tf is None:
        try:
            tf = __import__("tensorflow")
        except ImportError:
            tf = None

    if tf is None:
        raise BackendNotSupportedError("TensorFlow is not installed or available in this environment.")

    # Convert Python sequences to native TensorFlow tensor primitives where possible
    has_conv, conv_fn = adapt_backend_module(tf).resolve_symbol("convert_to_tensor")
    processed_args: list[BackendArg] = []
    for arg in args:
        if isinstance(arg, (list, tuple)) and has_conv and conv_fn is not None:
            try:
                processed_args.append(conv_fn(arg))
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

    schema = load_backend_mappings("tensorflow")
    if op_type in schema.operations:
        return dispatch_eager_op("tensorflow", op_type, processed_args, dict(kwargs), backend_module=tf)

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(tf, *args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not supported by tensorflow eager backend.")
