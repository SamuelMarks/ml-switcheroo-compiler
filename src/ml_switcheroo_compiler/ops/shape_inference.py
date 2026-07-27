"""Shape inference module."""

from typing import Callable

# Fallback to the OpDef's infer_shape method if it still has it

_SHAPE_INFERENCE_REGISTRY: dict[str, Callable] = {}


def register_shape_inference(op_type: str) -> Callable:
    """Decorator to register a shape inference function for an op."""

    def decorator(func: Callable) -> Callable:
        """Evaluate and process the decorator operation.

        Args:
            func (Callable): Required parameter for func.

        Returns:
            Callable: The evaluated or processed output.
        """
        _SHAPE_INFERENCE_REGISTRY[op_type] = func
        return func

    return decorator


def infer_shape(op_type: str, *args: object, **kwargs: object) -> object:
    """Infer shape using the registered function."""
    if op_type in _SHAPE_INFERENCE_REGISTRY:
        return _SHAPE_INFERENCE_REGISTRY[op_type](*args, **kwargs)

    from ml_switcheroo_compiler.ops.registry import get_op

    op_cls = get_op(op_type)
    if hasattr(op_cls, "infer_shape") and callable(op_cls.infer_shape):
        return op_cls().infer_shape(*args, **kwargs)
    return ()
