"""Activations."""

import uuid
from typing import Optional
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode
from ml_switcheroo.core.errors import UnimplementedMathError


def _emit_nn_node(
    op_type: str, inputs: list, attrs: dict, out_shape: tuple, out_dtype: DType
) -> Tensor:
    if not _tracer.is_tracing:
        raise RuntimeError(f"Cannot emit {op_type} node outside of a tracing context.")
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attrs,
        shape_metadata=out_shape,
    )
    _tracer.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
    return Tensor(data=proxy, shape=out_shape, dtype=out_dtype, device=inputs[0].device)


def relu(input: Tensor) -> Tensor:
    """relu"""
    if config.eager_mode:
        data = np.maximum(0, input.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_nn_node("Relu", [input], {}, input.shape, input.dtype)


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    """leaky_relu"""
    if config.eager_mode:
        data = np.where(input.data > 0, input.data, input.data * negative_slope)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_nn_node(
            "LeakyRelu",
            [input],
            {"negative_slope": negative_slope},
            input.shape,
            input.dtype,
        )


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    """gelu"""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for gelu")

    else:
        return _emit_nn_node(
            "Gelu", [input], {"approximate": approximate}, input.shape, input.dtype
        )


def swish(input: Tensor) -> Tensor:
    """swish"""
    if config.eager_mode:
        data = input.data / (1 + np.exp(-input.data))
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_nn_node("Swish", [input], {}, input.shape, input.dtype)


def sigmoid(input: Tensor) -> Tensor:
    """sigmoid"""
    if config.eager_mode:
        data = 1 / (1 + np.exp(-input.data))
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_nn_node("Sigmoid", [input], {}, input.shape, input.dtype)


def tanh(input: Tensor) -> Tensor:
    """tanh"""
    if config.eager_mode:
        data = np.tanh(input.data)
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_nn_node("Tanh", [input], {}, input.shape, input.dtype)


def softplus(input: Tensor, beta: float = 1, threshold: float = 20) -> Tensor:
    """softplus"""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for softplus")

    else:
        return _emit_nn_node(
            "Softplus",
            [input],
            {"beta": beta, "threshold": threshold},
            input.shape,
            input.dtype,
        )


def elu(input: Tensor, alpha: float = 1.0) -> Tensor:
    """elu"""
    if config.eager_mode:
        data = np.where(input.data > 0, input.data, alpha * (np.exp(input.data) - 1))
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_nn_node("Elu", [input], {"alpha": alpha}, input.shape, input.dtype)


def selu(input: Tensor) -> Tensor:
    """selu"""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for selu")

    else:
        return _emit_nn_node("Selu", [input], {}, input.shape, input.dtype)


def celu(input: Tensor, alpha: float = 1.0) -> Tensor:
    """celu"""
    if config.eager_mode:
        data = np.maximum(0, input.data) + np.minimum(
            0, alpha * (np.exp(input.data / alpha) - 1)
        )
        return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)
    else:
        return _emit_nn_node(
            "Celu", [input], {"alpha": alpha}, input.shape, input.dtype
        )


def glu(input: Tensor, dim: int = -1) -> Tensor:
    """glu"""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for glu")

    else:
        return _emit_nn_node("Glu", [input], {"dim": dim}, input.shape, input.dtype)


def mish(input: Tensor) -> Tensor:
    """mish"""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for mish")

    else:
        return _emit_nn_node("Mish", [input], {}, input.shape, input.dtype)


def hardswish(input: Tensor) -> Tensor:
    """hardswish"""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for hardswish")

    else:
        return _emit_nn_node("Hardswish", [input], {}, input.shape, input.dtype)


def softmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """softmax"""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for softmax")

    else:
        return _emit_nn_node("Softmax", [input], {"dim": dim}, input.shape, input.dtype)


def log_softmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """log_softmax"""
    if config.eager_mode:
        raise UnimplementedMathError("No direct numpy for log_softmax")

    else:
        return _emit_nn_node(
            "LogSoftmax", [input], {"dim": dim}, input.shape, input.dtype
        )
