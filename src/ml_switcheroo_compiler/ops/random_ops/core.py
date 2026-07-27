"""Core Random."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Rademacher")
class Rademacher(OpDef):
    """Draw samples from the Rademacher distribution."""

    op_name = "Rademacher"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        if "shape" in kwargs and kwargs["shape"] is not None:
            return tuple(kwargs["shape"]) if isinstance(kwargs["shape"], (list, tuple)) else (kwargs["shape"],)
        return ()
