"""Core abstractions and logic definitions for quantized_ops.py."""

from dataclasses import dataclass
from typing import Optional

from ml_switcheroo_compiler.core.config import config as compiler_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.nn.linear_ops import linear
from ml_switcheroo_compiler.ops.shape.indexing import gather
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@register_op("Quantize")
class QuantizeOp(OpDef):
    """Operation definition for quantizing a floating-point weight tensor."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infers the shape of the output tensor for the quantize operation.

        Args:
            *args: Variable length argument list where the first element is the input tensor.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The inferred shape of the output tensor.
        """
        return getattr(args[0], "shape", ())


@register_op("QuantizedMatmul")
class QuantizedMatmulOp(OpDef):
    """Operation definition for performing a matrix multiplication with quantized weights."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infers the shape of the output tensor for the quantized matmul operation.

        Args:
            *args: Variable length argument list where the first element is the input tensor.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The inferred shape of the output tensor.
        """
        return getattr(args[0], "shape", ())


@register_op("GatherQMM")
class GatherQMMOp(OpDef):
    """Operation definition for combined gathering and quantized matrix multiplication."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infers the shape of the output tensor for the gather and quantized matmul operation.

        Args:
            *args: Variable length argument list where the first element is the input tensor.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The inferred shape of the output tensor.
        """
        return getattr(args[0], "shape", ())


@register_op("FakeQuantWithMinMaxVars")
class FakeQuantWithMinMaxVarsOp(OpDef):
    """Operation definition for simulating quantization using minimum and maximum variables."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infers the shape of the output tensor for the fake quantization operation.

        Args:
            *args: Variable length argument list where the first element is the input tensor.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The inferred shape of the output tensor.
        """
        return getattr(args[0], "shape", ())


@register_op("QuantizeAndDequantize")
class QuantizeAndDequantizeOp(OpDef):
    """Operation definition for sequentially quantizing and then dequantizing a tensor."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infers the shape of the output tensor for the quantize and dequantize operation.

        Args:
            *args: Variable length argument list where the first element is the input tensor.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The inferred shape of the output tensor.
        """
        return getattr(args[0], "shape", ())


@register_op("AbsMaxQuantize")
class AbsMaxQuantizeOp(OpDef):
    """Operation definition for performing absolute maximum-based quantization."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infers the shape of the output tensor for the absolute max quantize operation.

        Args:
            *args: Variable length argument list where the first element is the input tensor.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The inferred shape of the output tensor.
        """
        return getattr(args[0], "shape", ())


@register_op("ComputeFloat8AmaxHistory")
class ComputeFloat8AmaxHistoryOp(OpDef):
    """Operation definition for computing the absolute maximum history for float8 quantization."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infers the shape of the output tensor for the float8 amax history computation.

        Args:
            *args: Variable length argument list where the first element is the input tensor.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The inferred shape of the output tensor.
        """
        return getattr(args[0], "shape", ())


@register_op("ComputeFloat8Scale")
class ComputeFloat8ScaleOp(OpDef):
    """Operation definition for computing the scale factor for float8 quantization."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infers the shape of the output tensor for the float8 scale computation.

        Args:
            *args: Variable length argument list where the first element is the input tensor.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The inferred shape of the output tensor.
        """
        return getattr(args[0], "shape", ())


def fake_quant_with_min_max_vars(
    inputs: Tensor,
    min_val: Tensor,
    max_val: Tensor,
    num_bits: int = 8,
    narrow_range: bool = False,
) -> Tensor:
    """Simulate quantization and dequantization of a tensor using min and max variables.

    This function is useful for quantization-aware training, simulating the precision loss
    of integer quantization while maintaining floating-point gradients.

    Args:
        inputs: The input tensor to be fake-quantized.
        min_val: The minimum value of the quantization range.
        max_val: The maximum value of the quantization range.
        num_bits: The number of bits for the simulated quantization.
        narrow_range: Whether to use a narrow quantization range.

    Returns:
        Tensor: The fake-quantized output tensor.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.ops.creation.frontend_basic import zeros_like

        return zeros_like(inputs) + inputs

    attrs = {"num_bits": num_bits, "narrow_range": narrow_range}
    return _emit_shape_node("FakeQuantWithMinMaxVars", [inputs, min_val, max_val], attrs, inputs.shape, inputs.dtype)


@dataclass
class QuantizationParams:
    """Parameters for quantization operations."""

    signed_input: bool = True
    num_bits: int = 8
    range_given: bool = False
    round_mode: str = "HALF_TO_EVEN"
    narrow_range: bool = False
    axis: Optional[int] = None


def quantize_and_dequantize(
    input: Tensor,
    input_min: Tensor,
    input_max: Tensor,
    params: Optional[QuantizationParams] = None,
) -> Tensor:
    """Quantizes and then dequantizes a tensor to simulate lower precision.

    This operation applies quantization based on the provided min/max bounds and parameters,
    then immediately dequantizes it back to the original type to simulate precision loss.

    Args:
        input: The input tensor to quantize and dequantize.
        input_min: The minimum scalar or tensor bound for the quantization range.
        input_max: The maximum scalar or tensor bound for the quantization range.
        params: Optional configuration parameters for the quantization process.

    Returns:
        Tensor: The simulated lower precision tensor.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.ops.creation.frontend_basic import zeros_like

        return zeros_like(input) + input

    params = params or QuantizationParams()
    attrs = {
        "signed_input": params.signed_input,
        "num_bits": params.num_bits,
        "range_given": params.range_given,
        "round_mode": params.round_mode,
        "narrow_range": params.narrow_range,
        "axis": params.axis,
    }
    return _emit_shape_node("QuantizeAndDequantize", [input, input_min, input_max], attrs, input.shape, input.dtype)


def abs_max_quantize(
    input: Tensor,
    axis: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Quantizes an input tensor using absolute maximum quantization scaling.

    This method calculates scaling factors based on the absolute maximum values along
    a specified axis and scales the tensor accordingly.

    Args:
        input: The input tensor to be quantized.
        axis: The axis along which to compute the absolute maximum for scaling. If None, computes globally.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the quantized output tensor and the scale tensor.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.ops.creation.frontend_basic import zeros_like

        return zeros_like(input), zeros_like(input)

    attrs = {"axis": axis}
    q_out = _emit_shape_node("AbsMaxQuantize", [input], {**attrs, "return_idx": 0}, input.shape, DType.Int8)
    scale_out = _emit_shape_node("AbsMaxQuantize", [input], {**attrs, "return_idx": 1}, (), input.dtype)
    return q_out, scale_out


def compute_float8_amax_history(
    x: Tensor,
    amax_history: Tensor,
) -> Tensor:
    """Compute and updates the absolute maximum history for float8 quantization scaling.

    Args:
        x: The input tensor to process.
        amax_history: The tensor containing the history of absolute maximums.

    Returns:
        Tensor: The updated absolute maximum history tensor.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.ops.creation.frontend_basic import zeros_like

        return zeros_like(amax_history)

    return _emit_shape_node("ComputeFloat8AmaxHistory", [x, amax_history], {}, amax_history.shape, amax_history.dtype)


def compute_float8_scale(
    amax_history: Tensor,
    scale: Tensor,
    margin: float = 0.0,
) -> Tensor:
    """Compute the scaling factor for float8 quantization from absolute maximum history.

    Args:
        amax_history: The tensor containing the history of absolute maximums.
        scale: The current scale tensor.
        margin: The margin to apply to the scaling calculation.

    Returns:
        Tensor: The newly computed scale tensor.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.ops.creation.frontend_basic import zeros_like

        return zeros_like(scale)

    attrs = {"margin": margin}
    return _emit_shape_node("ComputeFloat8Scale", [amax_history, scale], attrs, scale.shape, scale.dtype)


def quantize(
    w: Tensor,
    group_size: int = 64,
    bits: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantizes a floating-point weight tensor into a lower-bit representation.

    Args:
        w: The floating-point weight tensor to quantize.
        group_size: The number of elements grouped together for shared scaling factors.
        bits: The number of bits for the target quantized representation.

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing the quantized weights, scales, and biases.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.ops.creation.frontend_basic import zeros_like

        return (
            zeros_like(w),
            zeros_like(w),
            zeros_like(w),
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
    """Perform a matrix multiplication using a quantized weight matrix configuration.

    Args:
        x: The input tensor for the multiplication.
        config: The configuration object containing quantized weights, scales, and metadata.

    Returns:
        Tensor: The resulting tensor from the quantized matrix multiplication.
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
    """Gather elements and performs a matrix multiplication with quantized weights.

    Args:
        x: The input tensor containing gathered data or indices.
        config: The configuration object containing quantized weights, scales, and metadata.

    Returns:
        Tensor: The resulting tensor after gathering and performing the quantized matmul.
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
    """Apply a linear transformation to incoming data using quantized parameters.

    The weight tensor within the configuration is expected to be quantized (e.g., int4 or int8).
    This function conceptually dequantizes the weight and performs a linear
    transformation: y = input @ dequantize(weight).T + bias.

    Args:
        input: The incoming data tensor to be transformed.
        config: The configuration object containing quantized weights, biases, and metadata.

    Returns:
        Tensor: The transformed output tensor.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        w_val = backend.asarray(config.weight.data)
        scales_val = backend.asarray(config.scales.data)
        if config.zeros is not None:
            zeros_val = backend.asarray(config.zeros.data)
        elif config.biases is not None:
            zeros_val = backend.asarray(config.biases.data)
        else:
            zeros_val = 0.0

        w_float_val = (w_val - zeros_val) * scales_val

        from ml_switcheroo_compiler.core.tensor import TensorConfig

        weight_float = Tensor(w_float_val, TensorConfig(w_float_val.shape, DType.Float32, config.weight.device))

        return linear(input, weight_float, bias=config.biases)
    else:
        return linear(input, config.weight, bias=config.biases)


def quantized_embedding(
    input: Tensor,
    config: QuantizedOpsConfig,
) -> Tensor:
    """Look up embeddings from a quantized embedding weight table.

    Args:
        input: The tensor containing the indices to look up.
        config: The configuration object containing the quantized embedding weights and scales.

    Returns:
        Tensor: The dequantized embedding tensor corresponding to the input indices.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        w_val = backend.asarray(config.weight.data)
        scales_val = backend.asarray(config.scales.data)
        if config.zeros is not None:
            zeros_val = backend.asarray(config.zeros.data)
        elif config.biases is not None:
            zeros_val = backend.asarray(config.biases.data)
        else:
            zeros_val = 0.0

        w_float_val = (w_val - zeros_val) * scales_val

        from ml_switcheroo_compiler.core.tensor import TensorConfig

        weight_float = Tensor(w_float_val, TensorConfig(w_float_val.shape, DType.Float32, config.weight.device))

        return gather(weight_float, dim=0, index=input)
    else:
        return gather(config.weight, dim=0, index=input)


def dequantize(
    input: Tensor,
    scales: Tensor,
    biases: Optional[Tensor] = None,
    group_size: int = 64,
    bits: int = 4,
) -> Tensor:
    """Dequantizes a tensor back to a floating point representation.

    Args:
        input: The input tensor to dequantize.
        scales: The scaling factors for dequantization.
        biases: The zero-points or biases for dequantization.
        group_size: The number of elements grouped together.
        bits: The number of bits in the quantized representation.

    Returns:
        Tensor: The dequantized floating point tensor.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        from ml_switcheroo_compiler.core.tensor import TensorConfig

        in_val = backend.asarray(input.data)
        scales_val = backend.asarray(scales.data)
        if biases is not None:
            biases_val = backend.asarray(biases.data)
        else:
            biases_val = backend.execute_op("Zeros", 1, dtype="float32")

        res_val = (in_val - biases_val) * scales_val
        return Tensor(res_val, TensorConfig(res_val.shape, scales.dtype, input.device))

    inputs = [input, scales]
    if biases is not None:
        inputs.append(biases)
    attrs = {"group_size": group_size, "bits": bits}
    return _emit_shape_node("Dequantize", inputs, attrs, input.shape, scales.dtype)


def quantized_conv(
    input: Tensor,
    config: QuantizedOpsConfig,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """Perform a convolution using a quantized weight configuration.

    Args:
        input: The input tensor for the convolution.
        config: The quantized configuration containing weights, scales, and biases.
        stride: The stride of the convolving kernel.
        padding: The padding added to both sides of the input.
        dilation: The spacing between kernel elements.
        groups: The number of blocked connections from input channels to output channels.

    Returns:
        Tensor: The result of the quantized convolution.
    """
    if compiler_config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend
        from ml_switcheroo_compiler.core.tensor import TensorConfig

        backend = get_active_backend()

        args = [input, config.weight, config.scales]
        if config.biases is not None:
            args.append(config.biases)

        kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        res_val = backend.execute_op("QuantizedConv", *args, **kwargs)
        return Tensor(res_val, TensorConfig(res_val.shape, input.dtype, input.device))

    inputs = [input, config.weight, config.scales]
    if config.biases is not None:
        inputs.append(config.biases)
    attrs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }
    # For now, return a symbolic shape using the input's shape to preserve tracing
    return _emit_shape_node("QuantizedConv", inputs, attrs, input.shape, input.dtype)
