import os

os.makedirs("src/ml_switcheroo/nn", exist_ok=True)

# 1. Activations
activations = [
    "relu",
    "leaky_relu",
    "gelu",
    "swish",
    "sigmoid",
    "tanh",
    "softplus",
    "elu",
    "selu",
    "celu",
    "glu",
    "mish",
    "hardswish",
    "softmax",
    "log_softmax",
]

act_content = """\"\"\"Activations.\"\"\"
import uuid
from typing import Optional
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.config import config
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode
from ml_switcheroo.core.errors import UnimplementedMathError

def _emit_nn_node(op_type: str, inputs: list, attrs: dict, out_shape: tuple, out_dtype: DType) -> Tensor:
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
"""

for op in activations:
    if op == "leaky_relu":
        sig = "(input: Tensor, negative_slope: float = 0.01) -> Tensor:"
        eager = (
            "data = np.where(input.data > 0, input.data, input.data * negative_slope)"
        )
        attrs = "{'negative_slope': negative_slope}"
    elif op == "gelu":
        sig = "(input: Tensor, approximate: str = 'none') -> Tensor:"
        eager = "raise UnimplementedMathError('No direct numpy for gelu')"
        attrs = "{'approximate': approximate}"
    elif op == "softplus":
        sig = "(input: Tensor, beta: float = 1, threshold: float = 20) -> Tensor:"
        eager = "raise UnimplementedMathError('No direct numpy for softplus')"
        attrs = "{'beta': beta, 'threshold': threshold}"
    elif op == "elu":
        sig = "(input: Tensor, alpha: float = 1.0) -> Tensor:"
        eager = "data = np.where(input.data > 0, input.data, alpha * (np.exp(input.data) - 1))"
        attrs = "{'alpha': alpha}"
    elif op == "celu":
        sig = "(input: Tensor, alpha: float = 1.0) -> Tensor:"
        eager = "data = np.maximum(0, input.data) + np.minimum(0, alpha * (np.exp(input.data / alpha) - 1))"
        attrs = "{'alpha': alpha}"
    elif op == "glu":
        sig = "(input: Tensor, dim: int = -1) -> Tensor:"
        eager = "raise UnimplementedMathError('No direct numpy for glu')"
        attrs = "{'dim': dim}"
    elif op in ["softmax", "log_softmax"]:
        sig = "(input: Tensor, dim: Optional[int] = None) -> Tensor:"
        eager = f"raise UnimplementedMathError('No direct numpy for {op}')"
        attrs = "{'dim': dim}"
    else:
        sig = "(input: Tensor) -> Tensor:"
        attrs = "{}"
        if op == "relu":
            eager = "data = np.maximum(0, input.data)"
        elif op == "swish":
            eager = "data = input.data / (1 + np.exp(-input.data))"
        elif op == "sigmoid":
            eager = "data = 1 / (1 + np.exp(-input.data))"
        elif op == "tanh":
            eager = "data = np.tanh(input.data)"
        elif op == "selu":
            eager = "raise UnimplementedMathError('No direct numpy for selu')"
        elif op == "mish":
            eager = "raise UnimplementedMathError('No direct numpy for mish')"
        elif op == "hardswish":
            eager = "raise UnimplementedMathError('No direct numpy for hardswish')"

    op_type = "".join(x.title() for x in op.split("_"))
    act_content += f"""
def {op}{sig}
    \"\"\"{op}\"\"\"
    if config.eager_mode:
        {eager}
        {"return Tensor(np.array(data), np.array(data).shape, input.dtype, input.device)" if "raise" not in eager else ""}
    else:
        return _emit_nn_node('{op_type}', [input], {attrs}, input.shape, input.dtype)
"""

with open("src/ml_switcheroo/nn/activations.py", "w") as f:
    f.write(act_content)


# 2. Convolutions and Pooling and other complex ops
complex_ops = {
    "conv1d": "(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: Union[str, int] = 0, dilation: int = 1, groups: int = 1)",
    "conv2d": "(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[int, tuple] = 1, padding: Union[str, int, tuple] = 0, dilation: Union[int, tuple] = 1, groups: int = 1)",
    "conv3d": "(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[int, tuple] = 1, padding: Union[str, int, tuple] = 0, dilation: Union[int, tuple] = 1, groups: int = 1)",
    "conv_transpose1d": "(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, dilation: int = 1)",
    "conv_transpose2d": "(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, output_padding: Union[int, tuple] = 0, groups: int = 1, dilation: Union[int, tuple] = 1)",
    "conv_transpose3d": "(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, output_padding: Union[int, tuple] = 0, groups: int = 1, dilation: Union[int, tuple] = 1)",
    "max_pool1d": "(input: Tensor, kernel_size: int, stride: Optional[int] = None, padding: int = 0, dilation: int = 1, ceil_mode: bool = False)",
    "max_pool2d": "(input: Tensor, kernel_size: Union[int, tuple], stride: Optional[Union[int, tuple]] = None, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, ceil_mode: bool = False)",
    "max_pool3d": "(input: Tensor, kernel_size: Union[int, tuple], stride: Optional[Union[int, tuple]] = None, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, ceil_mode: bool = False)",
    "avg_pool1d": "(input: Tensor, kernel_size: int, stride: Optional[int] = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True)",
    "avg_pool2d": "(input: Tensor, kernel_size: Union[int, tuple], stride: Optional[Union[int, tuple]] = None, padding: Union[int, tuple] = 0, ceil_mode: bool = False, count_include_pad: bool = True)",
    "avg_pool3d": "(input: Tensor, kernel_size: Union[int, tuple], stride: Optional[Union[int, tuple]] = None, padding: Union[int, tuple] = 0, ceil_mode: bool = False, count_include_pad: bool = True)",
    "adaptive_avg_pool2d": "(input: Tensor, output_size: Union[int, tuple])",
    "fractional_max_pool2d": "(input: Tensor, kernel_size: Union[int, tuple], output_size: Optional[Union[int, tuple]] = None, output_ratio: Optional[Union[float, tuple]] = None)",
    "layer_norm": "(input: Tensor, normalized_shape: Sequence[int], weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1e-05)",
    "batch_norm": "(input: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor], weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-05)",
    "group_norm": "(input: Tensor, num_groups: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1e-05)",
    "rms_norm": "(input: Tensor, normalized_shape: Sequence[int], weight: Optional[Tensor] = None, eps: float = 1e-08)",
    "instance_norm": "(input: Tensor, running_mean: Optional[Tensor] = None, running_var: Optional[Tensor] = None, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, use_input_stats: bool = True, momentum: float = 0.1, eps: float = 1e-05)",
    "dropout": "(input: Tensor, p: float = 0.5, training: bool = True)",
    "alpha_dropout": "(input: Tensor, p: float = 0.5, training: bool = True)",
    "feature_alpha_dropout": "(input: Tensor, p: float = 0.5, training: bool = True)",
    "spatial_dropout": "(input: Tensor, p: float = 0.5, training: bool = True)",
    "embedding": "(input: Tensor, weight: Tensor, padding_idx: Optional[int] = None, max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False)",
    "pad": "(input: Tensor, pad: Sequence[int], mode: str = 'constant', value: float = 0.0)",
    "upsample_bilinear": "(input: Tensor, size: Optional[Union[int, tuple]] = None, scale_factor: Optional[Union[float, tuple]] = None)",
    "upsample_nearest": "(input: Tensor, size: Optional[Union[int, tuple]] = None, scale_factor: Optional[Union[float, tuple]] = None)",
    "scaled_dot_product_attention": "(query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False)",
    "rnn_cell": "(input: Tensor, hx: Tensor, weight_ih: Tensor, weight_hh: Tensor, bias_ih: Optional[Tensor] = None, bias_hh: Optional[Tensor] = None)",
    "lstm_cell": "(input: Tensor, hx: Tuple[Tensor, Tensor], weight_ih: Tensor, weight_hh: Tensor, bias_ih: Optional[Tensor] = None, bias_hh: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]",
    "gru_cell": "(input: Tensor, hx: Tensor, weight_ih: Tensor, weight_hh: Tensor, bias_ih: Optional[Tensor] = None, bias_hh: Optional[Tensor] = None)",
}

comp_content = """\"\"\"Complex NN Operations.\"\"\"
import uuid
from typing import Optional, Union, Tuple, Sequence
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.config import config
from ml_switcheroo.core.errors import UnimplementedMathError
from ml_switcheroo.nn.activations import _emit_nn_node
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode

"""

for op, sig in complex_ops.items():
    op_type = "".join(x.title() for x in op.split("_"))
    if op == "lstm_cell":
        # special return tuple
        comp_content += f"""
def {op}{sig}:
    \"\"\"{op}\"\"\"
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for {op}.")
    else:
        if not _tracer.is_tracing:
             raise RuntimeError("Cannot emit {op} node outside of tracing context.")
        out_id_1 = str(uuid.uuid4())
        out_id_2 = str(uuid.uuid4())
        inputs = [input.data.id, hx[0].data.id, hx[1].data.id, weight_ih.data.id, weight_hh.data.id]
        if bias_ih is not None: inputs.append(bias_ih.data.id)
        if bias_hh is not None: inputs.append(bias_hh.data.id)
        node = LogicalNode(
            id=out_id_1,
            op_type="{op_type}",
            inputs=inputs,
            attributes={{}},
            shape_metadata=input.shape,
        )
        _tracer.add_node(node)
        proxy1 = ProxyTensor(id=out_id_1, shape=input.shape, dtype=input.dtype.value)
        proxy2 = ProxyTensor(id=out_id_2, shape=input.shape, dtype=input.dtype.value)
        return (Tensor(data=proxy1, shape=input.shape, dtype=input.dtype, device=input.device),
                Tensor(data=proxy2, shape=input.shape, dtype=input.dtype, device=input.device))
"""
    else:
        comp_content += f"""
def {op}{sig} -> Tensor:
    \"\"\"{op}\"\"\"
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for {op}.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
            elif isinstance(v, tuple) and len(v) > 0 and isinstance(v[0], Tensor): # for hx in rnn
                inputs.extend(list(v))
        return _emit_nn_node('{op_type}', inputs, {{}}, input.shape, input.dtype)
"""

with open("src/ml_switcheroo/nn/complex.py", "w") as f:
    f.write(comp_content)
