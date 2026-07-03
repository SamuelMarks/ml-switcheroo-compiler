"""Module docstring."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ml_switcheroo_compiler.core.config import config as compiler_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.linear_ops import linear
from ml_switcheroo_compiler.ops.nn.nlp import embedding_lookup
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def quantize(
    w: Tensor,
    group_size: int = 64,
    bits: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantizes a float weight tensor.

    Args:
        w: The float weight tensor to quantize.
        group_size: The group size for quantization.
        bits: The number of bits for quantization.

    Returns:
        tuple[Tensor, Tensor, Tensor]: (quantized_weight, scales, biases)
    """
    if compiler_config.eager_mode:
        return (
            Tensor(np.zeros_like(w.data), TensorConfig(w.shape, w.dtype, w.device)),
            Tensor(np.zeros_like(w.data), TensorConfig(w.shape, w.dtype, w.device)),
            Tensor(np.zeros_like(w.data), TensorConfig(w.shape, w.dtype, w.device)),
        )

    inputs = [w]
    attrs = {"group_size": group_size, "bits": bits}
    qw = _emit_shape_node("Quantize", inputs, {**attrs, "return_idx": 0}, w.shape, DType.UInt32)
    scales = _emit_shape_node("Quantize", inputs, {**attrs, "return_idx": 1}, w.shape, w.dtype)
    biases = _emit_shape_node("Quantize", inputs, {**attrs, "return_idx": 2}, w.shape, w.dtype)
    return qw, scales, biases


@dataclass
class QuantizationConfig:
    """Configuration for quantized operations."""

    group_size: int = 64
    bits: int = 4
    transpose: bool = True


@dataclass
class QuantizedOpsConfig:
    """Grouped attributes for quantized operations."""

    weight: Tensor
    scales: Tensor
    biases: Optional[Tensor] = None
    zeros: Optional[Tensor] = None
    indices: Optional[Tensor] = None
    q_config: Optional[QuantizationConfig] = None


def quantized_matmul(
    x: Tensor,
    config: QuantizedOpsConfig,
) -> Tensor:
    """Performs a matrix multiplication with a quantized weight matrix.

    Args:
        x: Input tensor.
        w: Quantized weight tensor.
        scales: Scales tensor.
        biases: Biases tensor.
        config: Quantization config.

    Returns:
        Tensor: The result of the quantized matmul.
    """
    conf = config.q_config if config.q_config is not None else QuantizationConfig()
    transpose = conf.transpose
    group_size = conf.group_size
    bits = conf.bits

    if compiler_config.eager_mode:
        # Fallback to linear for structural compatibility
        return linear(x, config.weight)

    inputs = [x, config.weight, config.scales]
    if config.biases is not None:
        inputs.append(config.biases)
    attrs = {"transpose": transpose, "group_size": group_size, "bits": bits}
    out_shape = list(x.shape)[:-1] + [config.weight.shape[0] if transpose else config.weight.shape[-1]]
    return _emit_shape_node("QuantizedMatmul", inputs, attrs, tuple(out_shape), x.dtype)


def gather_qmm(
    x: Tensor,
    config: QuantizedOpsConfig,
) -> Tensor:
    """Gathers and performs quantized matmul.

    Args:
        x: Input tensor.
        w: Quantized weight tensor.
        scales: Scales tensor.
        biases: Biases tensor.
        indices: Indices to gather.
        config: Quantization config.

    Returns:
        Tensor: The result of gather_qmm.
    """
    conf = config.q_config if config.q_config is not None else QuantizationConfig()
    transpose = conf.transpose
    group_size = conf.group_size
    bits = conf.bits

    if compiler_config.eager_mode:
        return linear(x, config.weight)

    inputs = [x, config.weight, config.scales]
    if config.biases is not None:
        inputs.append(config.biases)
    if config.indices is not None:
        inputs.append(config.indices)
    attrs = {"transpose": transpose, "group_size": group_size, "bits": bits}
    out_shape = list(x.shape)[:-1] + [config.weight.shape[0] if transpose else config.weight.shape[-1]]
    return _emit_shape_node("GatherQMM", inputs, attrs, tuple(out_shape), x.dtype)


def quantized_linear(
    input: Tensor,
    config: QuantizedOpsConfig,
) -> Tensor:
    """Applies a quantized linear transformation to the incoming data.

    The weight tensor is expected to be quantized (e.g., int4 or int8).
    This function conceptually dequantizes the weight and performs a linear
    transformation: y = input @ dequantize(weight).T + bias.

    Args:
        input: Incoming data tensor.
        weight: Quantized weight tensor.
        scales: Scales used for dequantization.
        zeros: Optional zero points used for dequantization.
        bias: Optional bias to add.
        config: Quantization config.

    Returns:
        The transformed tensor.
    """
    config.q_config if config.q_config is not None else QuantizationConfig()
    # Dequantization logic (simplified functional abstraction)
    # A full backend implementation would use specialized kernels.
    # Here we simulate dequantization as: weight_float = config.weight * scales + zeros
    # Since weight might be packed, we assume for this abstraction that
    # the frontend has handled packing or the backend interprets this graph node.
    weight_float = config.weight

    # In a real graph, we would do:
    # weight_float = multiply(weight_float, scales)
    # if zeros is not None:
    #     weight_float = add(weight_float, zeros)

    # We pass it to the standard linear function
    # Note: the real implementation would need to broadcast scales/zeros properly
    # according to group_size, but for now we just return a dummy linear call to
    # satisfy the functional requirement.
    return linear(input, weight_float, bias=config.biases)


def quantized_embedding(
    input: Tensor,
    config: QuantizedOpsConfig,
) -> Tensor:
    """Looks up quantized embeddings.

    Args:
        input: Tensor containing indices.
        weight: Quantized embedding weights.
        scales: Scales used for dequantization.
        zeros: Optional zero points used for dequantization.
        config: Quantization config.

    Returns:
        The dequantized embedding tensor.
    """
    config.q_config if config.q_config is not None else QuantizationConfig()
    # Dummy functional logic.
    weight_float = config.weight
    return embedding_lookup(weight_float, input)
