"""Text and Categorical operations."""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, cast  # noqa: F401

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, EagerValue

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.registry import EagerValue
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config as global_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder


@dataclass
class AsStringConfig:
    """AsString Config."""

    precision: int = -1
    scientific: bool = False
    shortest: bool = False
    width: int = -1
    fill: str = ""


def string_to_hash(input_tensor: Tensor, num_buckets: int) -> EagerValue:
    """Hashes string tensors to integer buckets.

    Args:
        input_tensor (Tensor): Input string tensor.
        num_buckets (int): Number of hash buckets.

    Returns:
        Tensor: Hashed integers.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringToHash", input_tensor.data, num_buckets=num_buckets)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.Int32, input_tensor.device),
        )
    return _emit_shape_node(
        "StringToHash",
        [input_tensor],
        {"num_buckets": num_buckets},
        input_tensor.shape,
        DType.Int32,
    )


def regex_replace(input_tensor: Tensor, pattern: str, rewrite: str) -> EagerValue:
    """Replace matches of pattern in input_tensor with rewrite.

    Args:
        input_tensor (Tensor): Input string tensor.
        pattern (str): Regex pattern.
        rewrite (str): Rewrite string.

    Returns:
        Tensor: Replaced string tensor.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RegexReplace", input_tensor.data, pattern=pattern, rewrite=rewrite)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.String, input_tensor.device),
        )
    return _emit_shape_node(
        "RegexReplace",
        [input_tensor],
        {"pattern": pattern, "rewrite": rewrite},
        input_tensor.shape,
        DType.String,
    )


def regex_full_match(input_tensor: Tensor, pattern: str) -> EagerValue:
    """Check if each string fully matches the regex pattern.

    Args:
        input_tensor (Tensor): Input string tensor.
        pattern (str): Regex pattern.

    Returns:
        Tensor: Boolean tensor of matches.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("RegexFullMatch", input_tensor.data, pattern=pattern)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.Bool, input_tensor.device),
        )
    return _emit_shape_node(
        "RegexFullMatch",
        [input_tensor],
        {"pattern": pattern},
        input_tensor.shape,
        DType.Bool,
    )


def string_join(inputs: list[Tensor], separator: str = "") -> EagerValue:
    """Join strings in a list of tensors.

    Args:
        inputs (list[Tensor]): List of string tensors.
        separator (str): Separator to use.

    Returns:
        Tensor: Joined string tensor.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringJoin", [t.data for t in inputs], separator=separator)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.String, inputs[0].device),
        )
    return _emit_shape_node(
        "StringJoin",
        inputs,
        {"separator": separator},
        inputs[0].shape,
        DType.String,
    )


def string_length(input_tensor: Tensor) -> EagerValue:
    """Compute the length of each string.

    Args:
        input_tensor (Tensor): Input string tensor.

    Returns:
        Tensor: Lengths (Int32).
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringLength", input_tensor.data)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.Int32, input_tensor.device),
        )
    return _emit_shape_node(
        "StringLength",
        [input_tensor],
        {},
        input_tensor.shape,
        DType.Int32,
    )


def string_substr(input_tensor: Tensor, pos: int, len: int) -> EagerValue:
    """Return substrings.

    Args:
        input_tensor (Tensor): Input string tensor.
        pos (int): Starting position.
        len (int): Length of the substring.

    Returns:
        Tensor: Substrings.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringSubstr", input_tensor.data, pos=pos, len=len)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.String, input_tensor.device),
        )
    return _emit_shape_node(
        "StringSubstr",
        [input_tensor],
        {"pos": pos, "len": len},
        input_tensor.shape,
        DType.String,
    )


def _string_split_eager(input_tensor: Tensor, delimiter: str) -> EagerValue:
    """Evaluate _string_split_eager operation.

    Args:
        input_tensor (Tensor): The input_tensor parameter.
        delimiter (str): The delimiter parameter.

    Returns:
        tuple: Result.
    """
    backend = get_active_backend()
    tokens, lengths = cast("tuple[EagerValue, EagerValue]", backend.execute_op("StringSplit", input_tensor.data, delimiter=delimiter))
    tokens_arr = cast("TensorLike", backend.array(tokens))
    lengths_arr = cast("TensorLike", backend.array(lengths))
    return (
        Tensor(
            tokens_arr,
            TensorConfig(tokens_arr.shape, DType.String, input_tensor.device),
        ),
        Tensor(
            lengths_arr,
            TensorConfig(lengths_arr.shape, DType.Int32, input_tensor.device),
        ),
    )


def _string_split_trace(input_tensor: Tensor, delimiter: str) -> EagerValue:
    """Evaluate _string_split_trace operation.

    Args:
        input_tensor (Tensor): The input_tensor parameter.
        delimiter (str): The delimiter parameter.

    Returns:
        tuple: Result.

    Raises:
        RuntimeError: An exception.
    """
    if not global_tracing_state.is_tracing:
        raise RuntimeError("Cannot emit StringSplit node outside of a tracing context.")

    out_id_tokens = str(uuid.uuid4())
    out_id_lengths = str(uuid.uuid4())

    input_ids, _, _ = TracingNodeBuilder.extract_proxy_inputs((input_tensor,))

    node = cast("Callable[..., LogicalNode]", LogicalNode)(
        id=out_id_tokens,
        op_type="StringSplit",
        inputs=input_ids,
        attributes={"delimiter": delimiter, "secondary_id": out_id_lengths},
        shape_metadata=(),
    )
    global_tracing_state.add_node(node)

    proxy_tokens = ProxyTensor(id=out_id_tokens, shape=(), dtype="string")
    proxy_lengths = ProxyTensor(id=out_id_lengths, shape=(), dtype="int32")

    return (
        Tensor(proxy_tokens, TensorConfig((), DType.String, input_tensor.device)),
        Tensor(proxy_lengths, TensorConfig((), DType.Int32, input_tensor.device)),
    )


def string_split(input_tensor: Tensor, delimiter: str = " ") -> EagerValue:
    """Split string tensors into tokens.

    Args:
        input_tensor (Tensor): Input string tensor.
        delimiter (str): The delimiter.

    Returns:
        tuple[Tensor, Tensor]: Tokens and their lengths.
    """
    if global_config.eager_mode:
        return _string_split_eager(input_tensor, delimiter)
    return _string_split_trace(input_tensor, delimiter)


def lookup(input_tensor: Tensor, vocabulary: Tensor) -> EagerValue:
    """Map tensor values to integer indices using a vocabulary.

    Args:
        input_tensor (Tensor): Input tensor (string or int).
        vocabulary (Tensor): Vocabulary tensor.

    Returns:
        Tensor: Integer indices.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Lookup", input_tensor.data, vocabulary=vocabulary.data)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.Int32, input_tensor.device),
        )
    return _emit_shape_node(
        "Lookup",
        [input_tensor, vocabulary],
        {},
        input_tensor.shape,
        DType.Int32,
    )


def text_vectorization(input_tensor: Tensor, **kwargs: EagerValue) -> EagerValue:
    """Text vectorization.

    Args:
        input_tensor: Input tensor.
        **kwargs: Kwargs.

    Returns:
        Tensor.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("TextVectorization", input_tensor.data, **kwargs)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.Int32, input_tensor.device),
        )
    return _emit_shape_node(
        "TextVectorization",
        [input_tensor],
        kwargs,
        input_tensor.shape,
        DType.Int32,
    )


def string_to_number(input_tensor: Tensor, dtype: DType = DType.Float32) -> EagerValue:
    """Parse numeric values from string tensors.

    Args:
        input_tensor (Tensor): Input string tensor.
        dtype (DType): Target numeric type.

    Returns:
        Tensor: Parsed numeric tensor.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringToNumber", input_tensor.data, dtype=dtype)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, dtype, input_tensor.device),
        )
    return _emit_shape_node(
        "StringToNumber",
        [input_tensor],
        {"dtype": dtype},
        input_tensor.shape,
        dtype,
    )


def string_lower(input_tensor: Tensor) -> EagerValue:
    """Convert string tensors to lowercase.

    Args:
        input_tensor (Tensor): Input string tensor.

    Returns:
        Tensor: Lowercased string tensor.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringLower", input_tensor.data)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.String, input_tensor.device),
        )
    return _emit_shape_node(
        "StringLower",
        [input_tensor],
        {},
        input_tensor.shape,
        DType.String,
    )


def string_upper(input_tensor: Tensor) -> EagerValue:
    """Convert string tensors to uppercase.

    Args:
        input_tensor (Tensor): Input string tensor.

    Returns:
        Tensor: Uppercased string tensor.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringUpper", input_tensor.data)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.String, input_tensor.device),
        )
    return _emit_shape_node(
        "StringUpper",
        [input_tensor],
        {},
        input_tensor.shape,
        DType.String,
    )


def edit_distance(hypothesis: Tensor, truth: Tensor, normalize: bool = True) -> EagerValue:
    """Compute the Levenshtein distance between sequences.

    Args:
        hypothesis (Tensor): The hypothesis sequences.
        truth (Tensor): The truth sequences.
        normalize (bool): Whether to normalize the distance by truth length.

    Returns:
        Tensor: The edit distances.
    """
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("EditDistance", hypothesis.data, truth.data, normalize=normalize)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.Float32, hypothesis.device),
        )
    return _emit_shape_node(
        "EditDistance",
        [hypothesis, truth],
        {"normalize": normalize},
        hypothesis.shape,
        DType.Float32,
    )


def _as_string_config_to_dict(conf: Optional[AsStringConfig]) -> EagerValue:
    """Convert AsStringConfig to a dictionary.

    Args:
        conf (Any): The conf parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    conf = conf if conf is not None else AsStringConfig()
    return cast(
        EagerValue,
        {
            "precision": conf.precision,
            "scientific": conf.scientific,
            "shortest": conf.shortest,
            "width": conf.width,
            "fill": conf.fill,
        },
    )


def as_string(
    input_tensor: Tensor,
    config: Optional[AsStringConfig] = None,
) -> EagerValue:
    """Convert a numeric tensor to a string tensor.

    Args:
        input_tensor (Tensor): A numeric tensor.
        config (Optional[AsStringConfig]): Formatting configuration.

    Returns:
        Tensor: A string tensor of the same shape.
    """
    kwargs = _as_string_config_to_dict(config)
    if global_config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("AsString", input_tensor.data, **cast("dict[str, EagerValue]", kwargs))
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, DType.String, input_tensor.device),
        )
    return cast(
        EagerValue,
        cast("Callable[..., EagerTensorProtocol]", _emit_shape_node)(
            "AsString",
            [input_tensor],
            kwargs,
            input_tensor.shape,
            DType.String,
        ),
    )
