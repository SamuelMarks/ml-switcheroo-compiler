# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Text operations class definitions.

Provides operations and node definitions related to text manipulation and processing.
"""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import (
    Tensor,
    TensorConfig,
)
from ml_switcheroo_compiler.ops.base import (
    OpDef,
    get_op,
    register_op,
)


def _extract_shape(arg: object) -> tuple[int, ...]:
    """Extract shape tuple from tensor or shape metadata.

    Args:
        arg (object): Target tensor or shape structure.

    Returns:
        tuple[int, ...]: Extracted shape tuple.
    """
    if arg is None:
        return ()
    if hasattr(arg, "shape"):
        return tuple(int(d) for d in arg.shape)
    if hasattr(arg, "shape_metadata"):
        sm = arg.shape_metadata
        if sm:
            return tuple(int(d) for d in sm)
    if isinstance(arg, (tuple, list)):
        return tuple(int(d) for d in arg)
    return ()


@register_op("StringToHash")
class StringToHash(OpDef):
    """Operation that computes a hash value for a given string tensor."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:
        """Infer the output shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result shape matching input tensor.
        """
        inp = args[0] if args else kwargs.get("inputs", kwargs.get("input"))
        return _extract_shape(inp)


@register_op("RegexReplace")
class RegexReplace(OpDef):
    """Operation that replaces matches of a regular expression in a string tensor."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:
        """Infer the resulting shape after applying the regular expression replacement.

        Args:
            *args (object): Variable length argument list including input tensors.
            **kwargs (object): Arbitrary keyword arguments configuring the operation.

        Returns:
            tuple[int, ...]: Output shape matching input string tensor.
        """
        inp = args[0] if args else kwargs.get("input", kwargs.get("inputs"))
        return _extract_shape(inp)


@register_op("StringSplit")
class StringSplit(OpDef):
    """Operation that splits string elements based on a given delimiter."""

    def infer_shape(self, *args, **kwargs):
        """Infer the resulting shape after splitting strings in the input tensor.

        Args:
            *args (object): Variable length argument list including input tensors.
            **kwargs (object): Arbitrary keyword arguments configuring the operation.

        Returns:
            tuple: Output shape with trailing token dimension.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        inp = args[0] if args else kwargs.get("input", kwargs.get("inputs"))
        if inp is None:
            return ()
        s = _extract_shape(inp)
        if not s:
            return ()
        max_splits = kwargs.get("max_splits")
        token_dim = int(max_splits) + 1 if isinstance(max_splits, int) and max_splits > 0 else SymVar("num_tokens")
        return s + (token_dim,)


@register_op("Lookup")
class Lookup(OpDef):
    """Operation that retrieves values from a table based on given keys."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:
        """Infer the resulting shape after applying the table lookup.

        Args:
            *args (object): Variable length argument list including input tensors.
            **kwargs (object): Arbitrary keyword arguments configuring the operation.

        Returns:
            tuple[int, ...]: Output shape matching input keys.
        """
        inp = args[0] if args else kwargs.get("input_tensor", kwargs.get("inputs", kwargs.get("keys")))
        return _extract_shape(inp)


@register_op("Hashing")
class Hashing(OpDef):
    """Operation that applies a hashing algorithm to map inputs to bins."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:
        """Infer the resulting shape after applying the hashing operation.

        Args:
            *args (object): Variable length argument list including input tensors.
            **kwargs (object): Arbitrary keyword arguments configuring the operation.

        Returns:
            tuple[int, ...]: Output shape matching inputs.
        """
        inp = args[0] if args else kwargs.get("inputs", kwargs.get("input"))
        return _extract_shape(inp)


@register_op("StringLookup")
class StringLookup(OpDef):
    """Operation that translates strings into integer indices using a vocabulary."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:
        """Infer the resulting shape after applying the string lookup operation.

        Args:
            *args (object): Variable length argument list including input tensors.
            **kwargs (object): Arbitrary keyword arguments configuring the operation.

        Returns:
            tuple[int, ...]: Output shape matching inputs.
        """
        inp = args[0] if args else kwargs.get("inputs", kwargs.get("input"))
        return _extract_shape(inp)


@register_op("IntegerLookup")
class IntegerLookup(OpDef):
    """Operation that translates integer indices into other integers or strings using a vocabulary."""

    def infer_shape(self, *args, **kwargs):
        """Infer the resulting shape after applying the integer lookup operation.

        Args:
            *args: Variable length argument list including input tensors.
            **kwargs: Arbitrary keyword arguments configuring the operation.

        Returns:
            A tuple representing the output shape.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("TextVectorization")
class TextVectorization(OpDef):
    """Operation that maps text features to integer sequences or dense vectors."""

    def infer_shape(self, *args, **kwargs):
        """Infer the resulting shape after applying the text vectorization operation.

        Args:
            *args (object): Variable length argument list including input tensors.
            **kwargs (object): Arbitrary keyword arguments configuring the operation.

        Returns:
            tuple: Vectorized sequence or feature shape.
        """
        from ml_switcheroo_compiler.ir.shape_system import SymVar

        inp = args[0] if args else kwargs.get("inputs", kwargs.get("input"))
        s = _extract_shape(inp) if inp is not None else ()
        if not s:
            return ()
        output_sequence_length = kwargs.get("output_sequence_length")
        if output_sequence_length is not None:
            return s + (int(output_sequence_length),)
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None and kwargs.get("output_mode") in ("count", "tf_idf", "multi_hot"):
            return s[:-1] + (int(max_tokens),) if s else (int(max_tokens),)
        return s + (SymVar("seq_len"),)


@register_op("StringToNumber")
class StringToNumber(OpDef):
    """Operation that converts strings to numeric values."""

    def infer_shape(self, *args, **kwargs):
        """Infer the resulting shape after converting strings to numbers.

        Args:
            *args: Variable length argument list including input tensors.
            **kwargs: Arbitrary keyword arguments configuring the operation.

        Returns:
            A tuple representing the output shape.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("StringLower")
class StringLower(OpDef):
    """Operation that converts string characters to lowercase."""

    def infer_shape(self, *args, **kwargs):
        """Infer the resulting shape after applying the lowercase conversion.

        Args:
            *args: Variable length argument list including input tensors.
            **kwargs: Arbitrary keyword arguments configuring the operation.

        Returns:
            A tuple representing the output shape.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("StringUpper")
class StringUpper(OpDef):
    """Operation that converts string characters to uppercase."""

    def infer_shape(self, *args, **kwargs):
        """Infer the resulting shape after applying the uppercase conversion.

        Args:
            *args: Variable length argument list including input tensors.
            **kwargs: Arbitrary keyword arguments configuring the operation.

        Returns:
            A tuple representing the output shape.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("StringJoin")
class StringJoin(OpDef):
    """Operation that joins an iterable of strings into a single string using a separator."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:
        """Infer the resulting shape after joining the strings.

        Args:
            *args (object): Variable length argument list including input tensors.
            **kwargs (object): Arbitrary keyword arguments configuring the operation.

        Returns:
            tuple[int, ...]: Output shape after join.
        """
        inp = args[0] if args else kwargs.get("inputs", kwargs.get("input"))
        if inp is None:
            return ()
        if isinstance(inp, (list, tuple)):
            if inp:
                return _extract_shape(inp[0])
            return ()
        s = list(_extract_shape(inp))
        axis = kwargs.get("axis")
        if axis is not None and s:
            ax = int(axis)
            if ax < 0:
                ax += len(s)
            if 0 <= ax < len(s):
                s.pop(ax)
            return tuple(s)
        return tuple(s)


@register_op("StringLength")
class StringLength(OpDef):
    """Operation that computes the length of each string in the input tensor."""

    def infer_shape(self, *args, **kwargs):
        """Infer the resulting shape after computing the string lengths.

        Args:
            *args: Variable length argument list including input tensors.
            **kwargs: Arbitrary keyword arguments configuring the operation.

        Returns:
            A tuple representing the output shape.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("StringSubstr")
class StringSubstr(OpDef):
    """Operation that extracts substrings from a tensor of strings."""

    def infer_shape(self, *args, **kwargs):
        """Infer the resulting shape after extracting substrings.

        Args:
            *args: Variable length argument list including input tensors.
            **kwargs: Arbitrary keyword arguments configuring the operation.

        Returns:
            A tuple representing the output shape.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("RegexFullMatch")
class RegexFullMatch(OpDef):
    """Operation that checks if the input strings fully match a given regular expression."""

    def infer_shape(self, *args, **kwargs):
        """Infer the resulting shape after checking for full regular expression matches.

        Args:
            *args: Variable length argument list including input tensors.
            **kwargs: Arbitrary keyword arguments configuring the operation.

        Returns:
            A tuple representing the output shape.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def string_to_hash(inputs: Tensor, **kwargs) -> Tensor:
    """Apply a hashing algorithm to map input string tensors to hash representations.

    Args:
        inputs: The incoming string tensor to be hashed.
        **kwargs: Extra parameters configuring the hashing backend or operation.

    Returns:
        A tensor comprising the generated hashed values.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Hashing", inputs.data, **kwargs)

        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, inputs.dtype, inputs.device),
        )

    return get_op("Hashing")()(inputs, **kwargs)


def lookup(inputs: Tensor, **kwargs) -> Tensor:
    """Map the given input strings or numbers into corresponding vocabulary indices or strings.

    Args:
        inputs: The incoming tensor containing the elements to look up.
        **kwargs: Extra parameters configuring the lookup operation, like the vocabulary file or dictionary.

    Returns:
        A tensor holding the retrieved vocabulary values or indices.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringLookup", inputs.data, **kwargs)

        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, inputs.dtype, inputs.device),
        )

    return get_op("StringLookup")()(inputs, **kwargs)


def text_vectorization(inputs: Tensor, **kwargs) -> Tensor:
    """Transform textual data into numerical tensor sequences based on pre-computed vocabularies.

    Args:
        inputs: The incoming string tensor containing sentences or text snippets.
        **kwargs: Extra parameters specifying the vectorization details such as maximum length or vocabulary mappings.

    Returns:
        A numeric tensor containing the vectorized representations of the incoming text.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("TextVectorization", inputs.data, **kwargs)

        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, inputs.dtype, inputs.device),
        )

    return get_op("TextVectorization")()(inputs, **kwargs)


@register_op("EditDistance")
class EditDistance(OpDef):
    """Operation that computes the Levenshtein distance between two sequences."""

    def infer_shape(self, hypothesis, truth=None, **kwargs) -> tuple[int, ...]:
        """Infer the resulting shape after computing the edit distance.

        Args:
            hypothesis (object): The predicted or generated sequence tensor.
            truth (object): The ground truth or reference sequence tensor.
            **kwargs (object): Arbitrary keyword arguments configuring the distance calculation.

        Returns:
            tuple[int, ...]: Reduced output batch shape.
        """
        s = _extract_shape(hypothesis)
        return s[:-1] if len(s) >= 1 else ()


@register_op("AsString")
class AsString(OpDef):
    """Operation that converts elements of a tensor into string representations."""

    def infer_shape(self, input_tensor, **kwargs) -> tuple[int, ...]:
        """Infer the resulting shape after formatting the input elements as strings.

        Args:
            input_tensor (object): The tensor containing the elements to be cast to strings.
            **kwargs (object): Arbitrary keyword arguments for formatting the output strings.

        Returns:
            tuple[int, ...]: Output shape matching input tensor.
        """
        return _extract_shape(input_tensor)


@register_op("ArrayRepr")
class ArrayRepr(OpDef):
    """Operation that computes the string representation of an array."""

    op_name = "ArrayRepr"

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("ArrayStr")
class ArrayStr(OpDef):
    """Operation that computes the string representation of an array."""

    op_name = "ArrayStr"

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()
