"""Activations and advanced NN operations."""

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.nn.activations import relu, selu, elu, gelu


def softplus(x: Tensor) -> Tensor:
    """Computes the softplus activation function.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The result of the softplus activation.
    """
    from ml_switcheroo_compiler.ops.unary import log1p, exp

    return log1p(exp(x))


__all__ = ["relu", "selu", "elu", "gelu", "softplus"]
