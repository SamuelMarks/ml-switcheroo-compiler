"""Text operations class definitions."""

from ml_switcheroo_compiler.core.tensor import Tensor

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("StringToHash")
class StringToHash(OpDef):
    """StringToHash op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("RegexReplace")
class RegexReplace(OpDef):
    """RegexReplace op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("StringSplit")
class StringSplit(OpDef):
    """StringSplit op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("Lookup")
class Lookup(OpDef):
    """Lookup op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        # The first argument is input_tensor. The shape should be the same as input_tensor.
        if args and isinstance(args[0], Tensor):
            return args[0].shape
        return ()
