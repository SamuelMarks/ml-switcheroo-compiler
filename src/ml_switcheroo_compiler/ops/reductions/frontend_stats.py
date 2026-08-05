"""Frontend reductions ops."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import dispatch_eager

from .frontend_utils import _emit_reduction_node


@dispatch_eager("Psum")
def psum(x: Tensor, axis_name: str) -> Tensor:
    """Compute an all-reduce sum over the specified mapped axis.

    Args:
        x (Tensor): The x parameter.
        axis_name (str): The axis_name parameter.

    Returns:
        Tensor: Result.
    """
    return _emit_reduction_node("Psum", [x], {"axis_name": axis_name}, x.shape, x.dtype)


@dispatch_eager("Pmean")
def pmean(x: Tensor, axis_name: str) -> Tensor:
    """Compute an all-reduce mean over the specified mapped axis.

    Args:
        x (Tensor): The x parameter.
        axis_name (str): The axis_name parameter.

    Returns:
        Tensor: Result.
    """
    return _emit_reduction_node("Pmean", [x], {"axis_name": axis_name}, x.shape, x.dtype)


@dispatch_eager("ApproxMaxK")
def approx_max_k(operand: Tensor, k: int, reduction_dimension: int = -1, recall_target: float = 0.95) -> tuple[Tensor, Tensor]:
    """Compute approximate top-k max elements and their indices.

    Args:
        operand (Tensor): The input tensor
        k (int): Number of top elements to look for along the last dimension
        reduction_dimension (int): The dimension to reduce along
        recall_target (float): The target recall

    Returns:
        tuple[Tensor, Tensor]: A tuple of (values, indices)
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
def approx_min_k(operand: Tensor, k: int, reduction_dimension: int = -1, recall_target: float = 0.95) -> tuple[Tensor, Tensor]:
    """Compute approximate top-k min elements and their indices.

    Args:
        operand (Tensor): The input tensor
        k (int): Number of top elements to look for along the last dimension
        reduction_dimension (int): The dimension to reduce along
        recall_target (float): The target recall

    Returns:
        tuple[Tensor, Tensor]: A tuple of (values, indices)
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


def corrcoef(x: object, y: object = None, rowvar: bool = True, bias: object = None, ddof: object = None) -> Tensor:
    """Return Pearson product-moment correlation coefficients.

    Args:
        x (object): The x parameter.
        y (object): The y parameter.
        rowvar (bool): The rowvar parameter.
        bias (object): The bias parameter.
        ddof (object): The ddof parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Corrcoef",
            getattr(x, "data", x),
            getattr(y, "data", y) if y is not None else None,
            rowvar=rowvar,
            bias=bias,
            ddof=ddof,
        )
        return Tensor(data, TensorConfig(data.shape, "float32", getattr(x, "device", None)))
    return _emit_reduction_node(
        "Corrcoef",
        [x, y] if y is not None else [x],
        {"rowvar": rowvar, "bias": bias, "ddof": ddof},
        (None, None),
        "float32",
    )


def correlate(a: object, v: object, mode: str = "valid") -> Tensor:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        a (object): The a parameter.
        v (object): The v parameter.
        mode (str): The mode parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        data = get_active_backend().execute_op("Correlate", getattr(a, "data", a), getattr(v, "data", v), mode=mode)
        return Tensor(data, TensorConfig(data.shape, "float32", getattr(a, "device", None)))
    return _emit_reduction_node("Correlate", [a, v], {"mode": mode}, (None,), "float32")


def cov(
    m: object,
    y: object = None,
    **kwargs: object,
) -> Tensor:
    """Estimate a covariance matrix, given data and weights.

    Args:
        m (object): The m parameter.
        y (object): The y parameter.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.

    Raises:
        ValueError: An exception.
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

    if config.eager_mode:
        data = get_active_backend().execute_op(
            "Cov",
            getattr(m, "data", m),
            getattr(y, "data", y) if y is not None else None,
            rowvar=rowvar,
            bias=bias,
            ddof=ddof,
            fweights=fweights,
            aweights=aweights,
        )
        return Tensor(data, TensorConfig(data.shape, "float32", getattr(m, "device", None)))
    return _emit_reduction_node(
        "Cov",
        [m, y] if y is not None else [m],
        {"rowvar": rowvar, "bias": bias, "ddof": ddof, "fweights": fweights, "aweights": aweights},
        (None, None),
        "float32",
    )
