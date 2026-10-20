"""Dedicated eager execution handlers and dispatch for the Keras backend."""

from __future__ import annotations

import sys
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.mapping_loader import OpMappingSchema
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def _get_keras_module() -> EagerValue | None:
    """Safely obtain or import keras module.

    Returns:
        Optional[object]: Keras module if present, else None.
    """
    keras = sys.modules.get("keras")
    if keras is None:
        try:
            keras = __import__("keras")
        except ImportError:
            keras = None
    return keras


def _convert_arg(arg: EagerValue, keras: EagerValue | None) -> EagerValue:
    """Convert input sequence argument to Keras tensor if possible.

    Args:
        arg (EagerValue): Argument to convert.
        keras (Optional[object]): Loaded keras module.

    Returns:
        EagerValue: Converted tensor or original object.
    """
    ops_mod = getattr(keras, "ops", None) if keras is not None else None
    if isinstance(arg, (list, tuple)) and ops_mod is not None and hasattr(ops_mod, "convert_to_tensor"):
        try:
            return typing.cast(EagerValue, ops_mod.convert_to_tensor(arg))
        except Exception:
            return arg
    return arg


def _try_dispatch_mapped_op(
    op_spec: OpMappingSchema,
    keras: EagerValue | None,
    args: list[EagerValue],
    kwargs: dict[str, EagerValue],
) -> EagerValue | None:
    """Attempt to dispatch operation from mapping schema.

    Args:
        op_spec (OpMappingSchema): Operation specification.
        keras (Optional[object]): Keras module.
        args (list[object]): Processed arguments.
        kwargs (dict[str, object]): Keyword arguments.

    Returns:
        Optional[object]: Result if dispatched, else None.
    """
    from ml_switcheroo_compiler.backends.mapping_loader import resolve_target_api

    target_api = getattr(op_spec, "target_api", None)
    custom_code = getattr(op_spec, "custom_code", None)
    if not (target_api or custom_code):
        return None

    func = resolve_target_api(typing.cast("str", target_api), custom_code, keras)
    if func is None or not callable(func):
        return None

    kwarg_trans = getattr(op_spec, "kwarg_translations", {}) or {}
    translated_kwargs: dict[str, EagerValue] = {kwarg_trans.get(k, k): v for k, v in kwargs.items()}
    if getattr(op_spec, "is_method", False) and args:
        method = getattr(args[0], typing.cast("str", target_api), None)
        if method is not None and callable(method):
            return typing.cast(typing.Union[EagerValue, "None"], method(*args[1:], **translated_kwargs))
    return func(*args, **translated_kwargs)


def execute_op(
    cls: type,
    op_type: str,
    *args: EagerValue,
    **kwargs: EagerValue,
) -> EagerValue:
    """Execute operation eagerly on Keras backend with native argument validation.

    Args:
        cls (type): The caller class or context.
        op_type (str): The name of the operation.
        *args (EagerValue): Positional arguments for the op.
        **kwargs (EagerValue): Keyword arguments for the op.

    Returns:
        EagerValue: Result of native Keras evaluation.

    Raises:
        BackendNotSupportedError: If Keras is unavailable or op is unmapped.
    """
    keras = _get_keras_module()
    processed_args: list[EagerValue] = [_convert_arg(arg, keras) for arg in args]

    from ml_switcheroo_compiler.backends.mapping_loader import load_backend_mappings

    schema = load_backend_mappings("keras")
    if op_type in schema.operations:
        res = _try_dispatch_mapped_op(schema.operations[op_type], keras, processed_args, dict(kwargs))
        if res is not None:
            return res

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(keras, *args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not supported by keras eager backend.")
