"""Complex NN Operations."""

import uuid
from typing import Optional, Union
from collections.abc import Sequence
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.config import config
from ml_switcheroo.core.errors import UnimplementedMathError
from ml_switcheroo.nn.activations import _emit_nn_node
from ml_switcheroo.tracing import _tracer, ProxyTensor
from ml_switcheroo_ir import LogicalNode


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: Union[str, int] = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """conv1d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for conv1d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("Conv1D", inputs, {}, inputs[0].shape, inputs[0].dtype)


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[str, int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1,
) -> Tensor:
    """conv2d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for conv2d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("Conv2D", inputs, {}, inputs[0].shape, inputs[0].dtype)


def conv3d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[str, int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1,
) -> Tensor:
    """conv3d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for conv3d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("Conv3D", inputs, {}, inputs[0].shape, inputs[0].dtype)


def conv_transpose1d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    dilation: int = 1,
) -> Tensor:
    """conv_transpose1d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for conv_transpose1d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "ConvTranspose1D", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    output_padding: Union[int, tuple] = 0,
    groups: int = 1,
    dilation: Union[int, tuple] = 1,
) -> Tensor:
    """conv_transpose2d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for conv_transpose2d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "ConvTranspose2D", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def conv_transpose3d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[int, tuple] = 0,
    output_padding: Union[int, tuple] = 0,
    groups: int = 1,
    dilation: Union[int, tuple] = 1,
) -> Tensor:
    """conv_transpose3d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for conv_transpose3d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "ConvTranspose3D", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def max_pool1d(
    input: Tensor,
    kernel_size: int,
    stride: Optional[int] = None,
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
) -> Tensor:
    """max_pool1d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for max_pool1d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("MaxPool1D", inputs, {}, inputs[0].shape, inputs[0].dtype)


def max_pool2d(
    input: Tensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    ceil_mode: bool = False,
) -> Tensor:
    """max_pool2d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for max_pool2d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("MaxPool2D", inputs, {}, inputs[0].shape, inputs[0].dtype)


def max_pool3d(
    input: Tensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    ceil_mode: bool = False,
) -> Tensor:
    """max_pool3d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for max_pool3d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("MaxPool3D", inputs, {}, inputs[0].shape, inputs[0].dtype)


def avg_pool1d(
    input: Tensor,
    kernel_size: int,
    stride: Optional[int] = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """avg_pool1d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for avg_pool1d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("AvgPool1D", inputs, {}, inputs[0].shape, inputs[0].dtype)


def avg_pool2d(
    input: Tensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """avg_pool2d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for avg_pool2d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("AvgPool2D", inputs, {}, inputs[0].shape, inputs[0].dtype)


def avg_pool3d(
    input: Tensor,
    kernel_size: Union[int, tuple],
    stride: Optional[Union[int, tuple]] = None,
    padding: Union[int, tuple] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """avg_pool3d."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for avg_pool3d.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("AvgPool3D", inputs, {}, inputs[0].shape, inputs[0].dtype)


def adaptive_avg_pool2d(input: Tensor, output_size: Union[int, tuple]) -> Tensor:
    """adaptive_avg_pool2d."""
    if config.eager_mode:
        raise UnimplementedMathError(
            "No direct NumPy equivalent for adaptive_avg_pool2d."
        )
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "AdaptiveAvgPool2D", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def fractional_max_pool2d(
    input: Tensor,
    kernel_size: Union[int, tuple],
    output_size: Optional[Union[int, tuple]] = None,
    output_ratio: Optional[Union[float, tuple]] = None,
) -> Tensor:
    """fractional_max_pool2d."""
    if config.eager_mode:
        raise UnimplementedMathError(
            "No direct NumPy equivalent for fractional_max_pool2d."
        )
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "FractionalMaxPool2D", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def layer_norm(
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-05,
) -> Tensor:
    """layer_norm."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for layer_norm.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("LayerNorm", inputs, {}, inputs[0].shape, inputs[0].dtype)


def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> Tensor:
    """batch_norm."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for batch_norm.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("BatchNorm", inputs, {}, inputs[0].shape, inputs[0].dtype)


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-05,
) -> Tensor:
    """group_norm."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for group_norm.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("GroupNorm", inputs, {}, inputs[0].shape, inputs[0].dtype)


def rms_norm(
    input: Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[Tensor] = None,
    eps: float = 1e-08,
) -> Tensor:
    """rms_norm."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for rms_norm.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("RmsNorm", inputs, {}, inputs[0].shape, inputs[0].dtype)


def instance_norm(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-05,
) -> Tensor:
    """instance_norm."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for instance_norm.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "InstanceNorm", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for dropout.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("Dropout", inputs, {}, inputs[0].shape, inputs[0].dtype)


def alpha_dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """alpha_dropout."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for alpha_dropout.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "AlphaDropout", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def feature_alpha_dropout(
    input: Tensor, p: float = 0.5, training: bool = True
) -> Tensor:
    """feature_alpha_dropout."""
    if config.eager_mode:
        raise UnimplementedMathError(
            "No direct NumPy equivalent for feature_alpha_dropout."
        )
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "FeatureAlphaDropout", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def spatial_dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """spatial_dropout."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for spatial_dropout.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "SpatialDropout", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    """Embedding."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for embedding.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("Embedding", inputs, {}, inputs[0].shape, inputs[0].dtype)


def pad(
    input: Tensor, pad: Sequence[int], mode: str = "constant", value: float = 0.0
) -> Tensor:
    """Pad."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for pad.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("Pad", inputs, {}, inputs[0].shape, inputs[0].dtype)


def upsample_bilinear(
    input: Tensor,
    size: Optional[Union[int, tuple]] = None,
    scale_factor: Optional[Union[float, tuple]] = None,
) -> Tensor:
    """upsample_bilinear."""
    if config.eager_mode:
        raise UnimplementedMathError(
            "No direct NumPy equivalent for upsample_bilinear."
        )
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "UpsampleBilinear", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def upsample_nearest(
    input: Tensor,
    size: Optional[Union[int, tuple]] = None,
    scale_factor: Optional[Union[float, tuple]] = None,
) -> Tensor:
    """upsample_nearest."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for upsample_nearest.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "UpsampleNearest", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tensor:
    """scaled_dot_product_attention."""
    if config.eager_mode:
        raise UnimplementedMathError(
            "No direct NumPy equivalent for scaled_dot_product_attention."
        )
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node(
            "ScaledDotProductAttention", inputs, {}, inputs[0].shape, inputs[0].dtype
        )


def rnn_cell(
    input: Tensor,
    hx: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor] = None,
    bias_hh: Optional[Tensor] = None,
) -> Tensor:
    """rnn_cell."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for rnn_cell.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("RnnCell", inputs, {}, inputs[0].shape, inputs[0].dtype)


def lstm_cell(
    input: Tensor,
    hx: tuple[Tensor, Tensor],
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor] = None,
    bias_hh: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """lstm_cell."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for lstm_cell.")
    else:
        out_id_1 = str(uuid.uuid4())
        out_id_2 = str(uuid.uuid4())
        inputs = [
            input.data.id,
            hx[0].data.id,
            hx[1].data.id,
            weight_ih.data.id,
            weight_hh.data.id,
        ]
        if bias_ih is not None:
            inputs.append(bias_ih.data.id)
        if bias_hh is not None:
            inputs.append(bias_hh.data.id)
        node = LogicalNode(
            id=out_id_1,
            op_type="LstmCell",
            inputs=inputs,
            attributes={},
            shape_metadata=input.shape,
        )
        _tracer.add_node(node)
        proxy1 = ProxyTensor(id=out_id_1, shape=input.shape, dtype=input.dtype.value)
        proxy2 = ProxyTensor(id=out_id_2, shape=input.shape, dtype=input.dtype.value)
        return (
            Tensor(
                data=proxy1, shape=input.shape, dtype=input.dtype, device=input.device
            ),
            Tensor(
                data=proxy2, shape=input.shape, dtype=input.dtype, device=input.device
            ),
        )


def gru_cell(
    input: Tensor,
    hx: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Optional[Tensor] = None,
    bias_hh: Optional[Tensor] = None,
) -> Tensor:
    """gru_cell."""
    if config.eager_mode:
        raise UnimplementedMathError("No direct NumPy equivalent for gru_cell.")
    else:
        # Simplification: we gather all Tensor args as inputs
        inputs = []
        locs = locals()
        for _k, v in locs.items():
            if isinstance(v, Tensor):
                inputs.append(v)
        return _emit_nn_node("GruCell", inputs, {}, inputs[0].shape, inputs[0].dtype)
