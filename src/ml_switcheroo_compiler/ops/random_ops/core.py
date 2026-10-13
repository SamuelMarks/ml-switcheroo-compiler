# ruff: noqa: E402, E501, F403
"""Core Random."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Rademacher")
class Rademacher(OpDef):
    """Draw samples from the Rademacher distribution."""

    op_name = "Rademacher"

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
        s = shape if shape is not None else kwargs.get("size")
        if s is None:
            return ()
        if isinstance(s, int):
            return (s,)
        return tuple(s)


def rademacher(*args, **kwargs):
    """Draw samples from a Rademacher distribution.

    Args:
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Rademacher", *args, **kwargs)
