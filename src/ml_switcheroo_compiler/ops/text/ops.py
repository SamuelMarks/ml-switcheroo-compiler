"""Text operations class definitions."""

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.config import config
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
        if args and isinstance(args[0], Tensor):  # pragma: no branch
            return args[0]  # pragma: no cover
        return ()  # pragma: no cover


@register_op("Hashing")
class Hashing(OpDef):
    """Hashing op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0]


@register_op("StringLookup")
class StringLookup(OpDef):
    """StringLookup op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0]


@register_op("IntegerLookup")
class IntegerLookup(OpDef):
    """IntegerLookup op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0]


@register_op("TextVectorization")
class TextVectorization(OpDef):
    """TextVectorization op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0]


@register_op("StringToNumber")
class StringToNumber(OpDef):
    """StringToNumber op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("StringLower")
class StringLower(OpDef):
    """StringLower op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("StringUpper")
class StringUpper(OpDef):
    """StringUpper op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("StringJoin")
class StringJoin(OpDef):
    """StringJoin op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return ()


@register_op("StringLength")
class StringLength(OpDef):
    """StringLength op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("StringSubstr")
class StringSubstr(OpDef):
    """StringSubstr op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("RegexFullMatch")
class RegexFullMatch(OpDef):
    """RegexFullMatch op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def string_to_hash(inputs: "Tensor", **kwargs: object) -> "Tensor":
    """String to hash.

    Args:
    inputs: Input tensor.
    **kwargs: Kwargs.

    Returns:
    Tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("Hashing", inputs.data, **kwargs)  # pragma: no cover
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

        return Tensor(  # pragma: no cover
            backend.array(data),
            TensorConfig(backend.array(data).shape, inputs.dtype, inputs.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    return get_op("Hashing")()(inputs, **kwargs)  # pragma: no cover


def lookup(inputs: "Tensor", **kwargs: object) -> "Tensor":
    """Lookup operation.

    Args:
    inputs: Input tensor.
    **kwargs: Kwargs.

    Returns:
    Tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("StringLookup", inputs.data, **kwargs)  # pragma: no cover
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

        return Tensor(  # pragma: no cover
            backend.array(data),
            TensorConfig(backend.array(data).shape, inputs.dtype, inputs.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    return get_op("StringLookup")()(inputs, **kwargs)  # pragma: no cover


def text_vectorization(inputs: "Tensor", **kwargs: object) -> "Tensor":
    """Text vectorization operation.

    Args:
    inputs: Input tensor.
    **kwargs: Kwargs.

    Returns:
    Tensor.
    """
    if config.eager_mode:  # pragma: no cover
        from ml_switcheroo_compiler.backends.registry import get_active_backend  # pragma: no cover

        backend = get_active_backend()  # pragma: no cover
        data = backend.execute_op("TextVectorization", inputs.data, **kwargs)  # pragma: no cover
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover

        return Tensor(  # pragma: no cover
            backend.array(data),
            TensorConfig(backend.array(data).shape, inputs.dtype, inputs.device),
        )
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    return get_op("TextVectorization")()(inputs, **kwargs)  # pragma: no cover


@register_op("EditDistance")
class EditDistance(OpDef):
    """EditDistance op."""

    def infer_shape(self, hypothesis: object, truth: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(hypothesis, "shape", ())


@register_op("AsString")
class AsString(OpDef):
    """AsString op."""

    def infer_shape(self, input_tensor: object, **kwargs: object) -> object:
        """Infer shape."""
        return getattr(input_tensor, "shape", ())
