"""Backend utilities."""

import mlx.core as mx

from ml_switcheroo_compiler.backends.eager_utils import (
    execute_generic_op,
)


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute execute_op.

    Args:
        cls (Any): The class.
        op_type (Any): Argument op_type.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    return execute_generic_op(mx, op_type, *args, **kwargs)
