# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Shape inference module."""

from typing import Callable

# Fallback to the OpDef's infer_shape method if it still has it

_SHAPE_INFERENCE_REGISTRY: dict[str, Callable[..., object]] = {}


def register_shape_inference(op_type: str) -> Callable[..., object]:
    """Decorate to register a shape inference function for an op.

    Args:
        op_type (str): The op_type parameter.

    Returns:
        Callable: Result.
    """

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        """Evaluate decorator operation.

        Args:
            func (Callable): The func parameter.

        Returns:
            Callable: Result.
        """
        _SHAPE_INFERENCE_REGISTRY[op_type] = func
        return func

    return decorator


def infer_shape(op_type: str, *args: object, **kwargs: object) -> object:
    """Infer shape using the registered function.

    Args:
        op_type (str): The op_type parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    if op_type in _SHAPE_INFERENCE_REGISTRY:
        return _SHAPE_INFERENCE_REGISTRY[op_type](*args, **kwargs)

    from ml_switcheroo_compiler.ops.registry import get_op

    op_cls: object = get_op(op_type)
    if hasattr(op_cls, "infer_shape") and callable(op_cls.infer_shape):
        return op_cls().infer_shape(*args, **kwargs)
    return ()
