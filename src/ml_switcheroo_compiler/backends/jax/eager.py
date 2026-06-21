"""Backend utilities."""

import jax.numpy as jnp

from ml_switcheroo_compiler.backends.eager import execute_generic_op


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
    return execute_generic_op(jnp, op_type, *args, **kwargs)
