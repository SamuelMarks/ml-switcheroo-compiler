"""Shape inference module."""

from typing import Callable

# Fallback to the OpDef's infer_shape method if it still has it

_SHAPE_INFERENCE_REGISTRY: dict[str, Callable] = {}


def register_shape_inference(op_type: str) -> Callable:
    """Decorator to register a shape inference function for an op."""

    def decorator(func: Callable) -> Callable:  # pragma: no cover
        """Function docstring.

        Args:
        func: Arg.
        """
        _SHAPE_INFERENCE_REGISTRY[op_type] = func  # pragma: no cover
        return func  # pragma: no cover

    return decorator  # pragma: no cover


def infer_shape(op_type: str, *args: object, **kwargs: object) -> object:
    """Infer shape using the registered function."""
    if op_type in _SHAPE_INFERENCE_REGISTRY:  # pragma: no branch
        return _SHAPE_INFERENCE_REGISTRY[op_type](*args, **kwargs)  # pragma: no cover

    from ml_switcheroo_compiler.ops.registry import get_op

    op_cls = get_op(op_type)
    if hasattr(op_cls, "infer_shape") and callable(op_cls.infer_shape):  # pragma: no branch
        return op_cls().infer_shape(*args, **kwargs)
    return ()  # pragma: no cover
