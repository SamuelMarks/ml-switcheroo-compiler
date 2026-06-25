# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, D103, PLR0913
"""Activations and advanced NN operations."""

from ml_switcheroo_compiler.nn.activations import (
    elu,
    gelu,
    relu,
    selu,
    softplus,
    celu,
    glu,
    hard_shrink,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    hard_tanh,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    relu6,
    sigmoid,
    silu,
    soft_shrink,
    softmax,
    soft_sign as softsign,
    sparse_plus,
    sparse_sigmoid,
    sparsemax,
    squareplus,
    swish,
    tanh_shrink,
    threshold,
)


def crelu(features, axis=-1, name=None):
    # pragma: no cover
    """Computes Concatenated ReLU."""
    from ml_switcheroo_compiler.ops.shape.frontend import concatenate
    from ml_switcheroo_compiler.ops import negative

    return relu(concatenate([features, negative(features)], dim=axis))


def isotonic_regression(y, sample_weights=None, increasing=True, name=None):
    # pragma: no cover
    """Solves isotonic regression problems."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig(y.shape, "float32", "cpu")), Tensor(
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
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_shrink",
    "softmax",
    "softplus",
    "softsign",
    "sparse_plus",
    "sparse_sigmoid",
    "sparsemax",
    "squareplus",
    "swish",
    "tanh_shrink",
    "threshold",
]
