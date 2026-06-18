"""Text and Categorical operations."""

import uuid
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.core.dtype import DType


def string_to_hash(input_tensor: Tensor, num_buckets: int) -> Tensor:
    """Hashes string tensors to integer buckets.

    Args:
        input_tensor (Tensor): Input string tensor.
        num_buckets (int): Number of hash buckets.

    Returns:
        Tensor: Hashed integers.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("StringToHash", input_tensor.data, num_buckets=num_buckets)
        return Tensor(
            backend.array(data), backend.array(data).shape, DType.Int32, input_tensor.device
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
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "RegexReplace", input_tensor.data, pattern=pattern, rewrite=rewrite
        )
        return Tensor(
            backend.array(data), backend.array(data).shape, DType.String, input_tensor.device
        )
    return _emit_shape_node(
        "RegexReplace",
        [input_tensor],
        {"pattern": pattern, "rewrite": rewrite},
        input_tensor.shape,
        DType.String,
    )


def _string_split_eager(input_tensor: Tensor, delimiter: str) -> tuple[Tensor, Tensor]:
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    tokens, lengths = backend.execute_op("StringSplit", input_tensor.data, delimiter=delimiter)
    return (
        Tensor(
            backend.array(tokens),
            backend.array(tokens).shape,
            DType.String,
            input_tensor.device,
        ),
        Tensor(
            backend.array(lengths),
            backend.array(lengths).shape,
            DType.Int32,
            input_tensor.device,
        ),
    )


def _string_split_trace(input_tensor: Tensor, delimiter: str) -> tuple[Tensor, Tensor]:
    from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor

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
        Tensor(proxy_tokens, (), DType.String, input_tensor.device),
        Tensor(proxy_lengths, (), DType.Int32, input_tensor.device),
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
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Lookup", input_tensor.data, vocabulary=vocabulary.data)
        return Tensor(
            backend.array(data), backend.array(data).shape, DType.Int32, input_tensor.device
        )
    return _emit_shape_node(
        "Lookup",
        [input_tensor, vocabulary],
        {},
        input_tensor.shape,
        DType.Int32,
    )
