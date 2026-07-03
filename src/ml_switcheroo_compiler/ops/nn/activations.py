"""Activations and advanced NN operations."""

# Dummy mock
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig  # pragma: no cover
from ml_switcheroo_compiler.nn.activations import (
    celu,
    elu,
    gelu,
    glu,
    hard_shrink,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    hard_tanh,
    hardshrink,
    hardswish,
    hardtanh,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    logsigmoid,
    mish,
    prelu,
    relu,
    relu2,
    relu6,
    selu,
    sigmoid,
    silu,
    soft_shrink,
    softmax,
    softmin,
    softplus,
    softshrink,
    sparse_plus,
    sparse_sigmoid,
    sparsemax,
    squareplus,
    step,
    swish,
    tanh_shrink,
    threshold,
)
from ml_switcheroo_compiler.nn.activations import (
    soft_sign as softsign,
)
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape.frontend import concatenate  # pragma: no cover


def crelu(features: object, axis: object = -1, name: object = None) -> object:
    # pragma: no cover
    """Computes Concatenated ReLU."""
    negative = get_op("Negative")()  # pragma: no cover

    # pragma: no cover
    return relu(concatenate([features, negative(features)], dim=axis))  # pragma: no cover


def isotonic_regression(y: object, sample_weights: object = None, increasing: object = True, name: object = None) -> object:
    # pragma: no cover
    """Solves isotonic regression problems."""
    # pragma: no cover
    return Tensor(None, TensorConfig(y.shape, "float32", "cpu")), Tensor(  # pragma: no cover
        None, TensorConfig(y.shape, "int32", "cpu")
    )


__all__ = [
    "celu",
    "elu",
    "gelu",
    "glu",
    "hard_shrink",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "hardshrink",
    "hardswish",
    "hardtanh",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "logsigmoid",
    "mish",
    "prelu",
    "relu",
    "relu2",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_shrink",
    "softmax",
    "softmin",
    "softplus",
    "softshrink",
    "softsign",
    "sparse_plus",
    "sparse_sigmoid",
    "sparsemax",
    "squareplus",
    "step",
    "swish",
    "tanh_shrink",
    "threshold",
]
