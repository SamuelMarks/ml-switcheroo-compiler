"""Text and Categorical operations."""

from ml_switcheroo_compiler.backends.registry import get_active_backend
import uuid

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def string_to_hash(input_tensor: Tensor, num_buckets: int) -> Tensor:
    """Hashes string tensors to integer buckets.

    Args:
        input_tensor (Tensor): Input string tensor.
        num_buckets (int): Number of hash buckets.

    Returns:
        Tensor: Hashed integers.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringToHash", input_tensor.data, num_buckets=num_buckets)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Int32, input_tensor.device),
        )
    return _emit_shape_node(
        "StringToHash",
        [input_tensor],
        {"num_buckets": num_buckets},
        input_tensor.shape,
        DType.Int32,
    )


def regex_replace(input_tensor: Tensor, pattern: str, rewrite: str) -> Tensor:
    """Replaces matches of pattern in input_tensor with rewrite.

    Args:
        input_tensor (Tensor): Input string tensor.
        pattern (str): Regex pattern.
        rewrite (str): Rewrite string.

    Returns:
        Tensor: Replaced string tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "RegexReplace", input_tensor.data, pattern=pattern, rewrite=rewrite
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.String, input_tensor.device),
        )
    return _emit_shape_node(
        "RegexReplace",
        [input_tensor],
        {"pattern": pattern, "rewrite": rewrite},
        input_tensor.shape,
        DType.String,
    )


def _string_split_eager(input_tensor: Tensor, delimiter: str) -> tuple[Tensor, Tensor]:

    backend = get_active_backend()
    tokens, lengths = backend.execute_op("StringSplit", input_tensor.data, delimiter=delimiter)
    return (
        Tensor(
            backend.array(tokens),
            TensorConfig(backend.array(tokens).shape, DType.String, input_tensor.device),
        ),
        Tensor(
            backend.array(lengths),
            TensorConfig(backend.array(lengths).shape, DType.Int32, input_tensor.device),
        ),
    )


def _string_split_trace(input_tensor: Tensor, delimiter: str) -> tuple[Tensor, Tensor]:
    from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer

    if not _tracer.is_tracing:
        raise RuntimeError("Cannot emit StringSplit node outside of a tracing context.")

    out_id_tokens = str(uuid.uuid4())
    out_id_lengths = str(uuid.uuid4())

    from ml_switcheroo_compiler.ops.base import get_op

    op_def = get_op("StringSplit")()
    input_ids, _, _ = op_def._extract_proxy_inputs((input_tensor,))

    from ml_switcheroo_ir import LogicalNode

    node = LogicalNode(
        id=out_id_tokens,
        op_type="StringSplit",
        inputs=input_ids,
        attributes={"delimiter": delimiter, "secondary_id": out_id_lengths},
        shape_metadata=(),
    )
    _tracer.add_node(node)

    proxy_tokens = ProxyTensor(id=out_id_tokens, shape=(), dtype="string")
    proxy_lengths = ProxyTensor(id=out_id_lengths, shape=(), dtype="int32")

    return (
        Tensor(proxy_tokens, TensorConfig((), DType.String, input_tensor.device)),
        Tensor(proxy_lengths, TensorConfig((), DType.Int32, input_tensor.device)),
    )


def string_split(input_tensor: Tensor, delimiter: str = " ") -> tuple[Tensor, Tensor]:
    """Splits string tensors into tokens.

    Args:
        input_tensor (Tensor): Input string tensor.
        delimiter (str): The delimiter.

    Returns:
        tuple[Tensor, Tensor]: Tokens and their lengths.
    """
    if config.eager_mode:
        return _string_split_eager(input_tensor, delimiter)
    return _string_split_trace(input_tensor, delimiter)


def lookup(input_tensor: Tensor, vocabulary: Tensor) -> Tensor:
    """Maps tensor values to integer indices using a vocabulary.

    Args:
        input_tensor (Tensor): Input tensor (string or int).
        vocabulary (Tensor): Vocabulary tensor.

    Returns:
        Tensor: Integer indices.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Lookup", input_tensor.data, vocabulary=vocabulary.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Int32, input_tensor.device),
        )
    return _emit_shape_node(
        "Lookup",
        [input_tensor, vocabulary],
        {},
        input_tensor.shape,
        DType.Int32,
    )


def text_vectorization(input_tensor: Tensor, **kwargs: object) -> Tensor:
    """Text vectorization.

    Args:
        input_tensor: Input tensor.
        **kwargs: Kwargs.

    Returns:
        Tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("TextVectorization", input_tensor.data, **kwargs)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Int32, input_tensor.device),
        )
    return _emit_shape_node(
        "TextVectorization",
        [input_tensor],
        kwargs,
        input_tensor.shape,
        DType.Int32,
    )


def string_to_number(input_tensor: Tensor, dtype: DType = DType.Float32) -> Tensor:
    """Parses numeric values from string tensors.

    Args:
        input_tensor (Tensor): Input string tensor.
        dtype (DType): Target numeric type.

    Returns:
        Tensor: Parsed numeric tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringToNumber", input_tensor.data, dtype=dtype)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, dtype, input_tensor.device),
        )
    return _emit_shape_node(
        "StringToNumber",
        [input_tensor],
        {"dtype": dtype},
        input_tensor.shape,
        dtype,
    )


def string_lower(input_tensor: Tensor) -> Tensor:
    """Converts string tensors to lowercase.

    Args:
        input_tensor (Tensor): Input string tensor.

    Returns:
        Tensor: Lowercased string tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringLower", input_tensor.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.String, input_tensor.device),
        )
    return _emit_shape_node(
        "StringLower",
        [input_tensor],
        {},
        input_tensor.shape,
        DType.String,
    )


def string_upper(input_tensor: Tensor) -> Tensor:
    """Converts string tensors to uppercase.

    Args:
        input_tensor (Tensor): Input string tensor.

    Returns:
        Tensor: Uppercased string tensor.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("StringUpper", input_tensor.data)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.String, input_tensor.device),
        )
    return _emit_shape_node(
        "StringUpper",
        [input_tensor],
        {},
        input_tensor.shape,
        DType.String,
    )


def edit_distance(hypothesis: Tensor, truth: Tensor, normalize: bool = True) -> Tensor:
    """Computes the Levenshtein distance between sequences.

    Args:
        hypothesis (Tensor): The hypothesis sequences.
        truth (Tensor): The truth sequences.
        normalize (bool): Whether to normalize the distance by truth length.

    Returns:
        Tensor: The edit distances.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("EditDistance", hypothesis.data, truth.data, normalize=normalize)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.Float32, hypothesis.device),
        )
    return _emit_shape_node(
        "EditDistance",
        [hypothesis, truth],
        {"normalize": normalize},
        hypothesis.shape,
        DType.Float32,
    )


def as_string(
    input_tensor: Tensor,
    precision: int = -1,
    scientific: bool = False,
    shortest: bool = False,
    width: int = -1,
    fill: str = "",
) -> Tensor:
    """Converts a numeric tensor to a string tensor.

    Args:
        input_tensor (Tensor): A numeric tensor.
        precision (int): The post-decimal precision to use.
        scientific (bool): Whether to use scientific notation.
        shortest (bool): Whether to use the shortest representation.
        width (int): The width to pad the output to.
        fill (str): The padding character.

    Returns:
        Tensor: A string tensor of the same shape.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "AsString",
            input_tensor.data,
            precision=precision,
            scientific=scientific,
            shortest=shortest,
            width=width,
            fill=fill,
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, DType.String, input_tensor.device),
        )
    return _emit_shape_node(
        "AsString",
        [input_tensor],
        {
            "precision": precision,
            "scientific": scientific,
            "shortest": shortest,
            "width": width,
            "fill": fill,
        },
        input_tensor.shape,
        DType.String,
    )
