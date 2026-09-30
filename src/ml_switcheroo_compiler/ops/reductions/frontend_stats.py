"""Module frontend_stats.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Frontend reductions ops."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import dispatch_eager

from .frontend_utils import _emit_reduction_node


@dispatch_eager("Psum")
def psum(x: Tensor, axis_name: str) -> Tensor:
    """Compute an all-reduce sum over the specified mapped axis.

    Args:
        x (Tensor): The input tensor to sum.
        axis_name (str): The named axis along which to reduce.

    Returns:
        Tensor: Reduced tensor result.
    """
    return _emit_reduction_node("Psum", [x], {"axis_name": axis_name}, x.shape, x.dtype)


@dispatch_eager("Pmean")
def pmean(x: Tensor, axis_name: str) -> Tensor:
    """Compute an all-reduce mean over the specified mapped axis.

    Args:
        x (Tensor): The input tensor to average.
        axis_name (str): The named axis along which to reduce.

    Returns:
        Tensor: Reduced tensor result.
    """
    return _emit_reduction_node("Pmean", [x], {"axis_name": axis_name}, x.shape, x.dtype)


@dispatch_eager("ApproxMaxK")
def approx_max_k(
    operand: Tensor,
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute approximate top-k max elements and their indices.

    Args:
        operand (Tensor): The input tensor.
        k (int): Number of top elements to look for along the last dimension.
        reduction_dimension (int): The dimension to reduce along.
        recall_target (float): The target recall.

    Returns:
        tuple[Tensor, Tensor]: A tuple of (values, indices).
    """
    attributes = {
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
    }

    val = _emit_reduction_node("ApproxMaxK", [operand], attributes, (), operand.dtype)
    idx = _emit_reduction_node("ApproxMaxKIndices", [operand], attributes, (), DType.Int32)
    return val, idx


@dispatch_eager("ApproxMinK")
def approx_min_k(
    operand: Tensor,
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute approximate top-k min elements and their indices.

    Args:
        operand (Tensor): The input tensor.
        k (int): Number of top elements to look for along the last dimension.
        reduction_dimension (int): The dimension to reduce along.
        recall_target (float): The target recall.

    Returns:
        tuple[Tensor, Tensor]: A tuple of (values, indices).
    """
    attributes = {
        "k": k,
        "reduction_dimension": reduction_dimension,
        "recall_target": recall_target,
    }

    val = _emit_reduction_node("ApproxMinK", [operand], attributes, (), operand.dtype)
    idx = _emit_reduction_node("ApproxMinKIndices", [operand], attributes, (), DType.Int32)
    return val, idx


def ctc_loss(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
) -> Tensor:
    """Connectionist Temporal Classification Loss.

    Args:
        log_probs (Tensor): Log probabilities.
        targets (Tensor): Targets.
        input_lengths (Tensor): Input lengths.
        target_lengths (Tensor): Target lengths.

    Returns:
        Tensor: The loss.
    """
    inputs = [log_probs, targets, input_lengths, target_lengths]
    return _emit_reduction_node("CTCLoss", inputs, {}, (), log_probs.dtype)


@dispatch_eager("Corrcoef")
def corrcoef(
    x: Tensor,
    y: Tensor | None = None,
    rowvar: bool = True,
    bias: bool | None = None,
    ddof: int | None = None,
) -> Tensor:
    """Return Pearson product-moment correlation coefficients.

    Args:
        x (Tensor): Input tensor.
        y (Tensor | None): Optional second input tensor.
        rowvar (bool): If True, each row represents a variable.
        bias (bool | None): Default normalization is False.
        ddof (int | None): Degrees of freedom.

    Returns:
        Tensor: Correlation matrix.
    """
    return _emit_reduction_node(
        "Corrcoef",
        [x, y] if y is not None else [x],
        {"rowvar": rowvar, "bias": bias, "ddof": ddof},
        (None, None),
        "float32",
    )


@dispatch_eager("Correlate")
def correlate(
    a: Tensor,
    v: Tensor,
    mode: str = "valid",
) -> Tensor:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        a (Tensor): First 1-D sequence.
        v (Tensor): Second 1-D sequence.
        mode (str): Mode of correlation ('valid', 'same', 'full').

    Returns:
        Tensor: Cross-correlation result.
    """
    return _emit_reduction_node("Correlate", [a, v], {"mode": mode}, (None,), "float32")


@dispatch_eager("Cov")
def cov(
    m: Tensor,
    y: Tensor | None = None,
    **kwargs: bool | int | float | Tensor | None,
) -> Tensor:
    """Estimate a covariance matrix, given data and weights.

    Args:
        m (Tensor): Input tensor.
        y (Tensor | None): Optional additional data.
        **kwargs (bool | int | float | Tensor | None): Keyword args including rowvar, bias, ddof, fweights, aweights.

    Returns:
        Tensor: Covariance matrix result.

    Raises:
        ValueError: If an unexpected keyword argument is provided.
    """
    allowed_keys = {"rowvar", "bias", "ddof", "fweights", "aweights"}
    for k in kwargs:
        if k not in allowed_keys:
            raise ValueError(f"Invalid keyword argument to cov: {k}")

    rowvar = kwargs.get("rowvar", True)
    bias = kwargs.get("bias", False)
    ddof = kwargs.get("ddof", None)
    fweights = kwargs.get("fweights", None)
    aweights = kwargs.get("aweights", None)

    return _emit_reduction_node(
        "Cov",
        [m, y] if y is not None else [m],
        {"rowvar": rowvar, "bias": bias, "ddof": ddof, "fweights": fweights, "aweights": aweights},
        (None, None),
        "float32",
    )
