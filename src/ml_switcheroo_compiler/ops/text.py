"""Text operations."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("CreateToken")
class CreateToken(OpDef):
    """CreateToken operation."""

    op_name = "CreateToken"


def create_token(*args: object, **kwargs: object) -> object:
    """Create token.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("CreateToken")()(*args, **kwargs)
