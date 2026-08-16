# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Calculate loss operations."""

from typing import Any

from ml_switcheroo_compiler.core.tensor import Tensor

# y_true * (log(y_true) - y_pred)
# usually implemented as y_true * log(y_true) - y_true * y_pred
from ml_switcheroo_compiler.ops.binary import (
    add,
    divide,
    greater,
    less_equal,
    maximum,
    multiply,
    subtract,
)
from ml_switcheroo_compiler.ops.creation.frontend import ones_like
from ml_switcheroo_compiler.ops.nn.activations import softplus
from ml_switcheroo_compiler.ops.nn.normalization import l2_normalize
from ml_switcheroo_compiler.ops.reductions import mean, sum
from ml_switcheroo_compiler.ops.shape.frontend import expand_dims
from ml_switcheroo_compiler.ops.shape.indexing import take_along_axis, where
from ml_switcheroo_compiler.ops.unary import abs, log, square


def l1_loss(y_true: Tensor, y_pred: Tensor) -> Any:  # type: ignore
    """L1 Loss.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.

    Returns:
        Tensor: Result.
    """
    return mean(abs(subtract(y_true, y_pred)))


def mse_loss(y_true: Tensor, y_pred: Tensor) -> Any:  # type: ignore
    """Mean Squared Error Loss.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.

    Returns:
        Tensor: Result.
    """
    return mean(square(subtract(y_true, y_pred)))


def huber_loss(y_true: Tensor, y_pred: Tensor, delta: float = 1.0) -> Any:  # type: ignore
    """Huber Loss.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.
        delta (float): The delta parameter.

    Returns:
        Tensor: Result.
    """
    error = subtract(y_true, y_pred)
    abs_error = abs(error)

    # 0.5 * x^2
    quadratic = multiply(0.5, square(error))
    # delta * (|x| - 0.5 * delta)
    linear = multiply(delta, subtract(abs_error, 0.5 * delta))

    loss = where(less_equal(abs_error, delta), quadratic, linear)
    return mean(loss)


def smooth_l1_loss(y_true: Tensor, y_pred: Tensor, beta: float = 1.0) -> Any:  # type: ignore
    """Smooth L1 Loss (similar to Huber with beta).

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.
        beta (float): The beta parameter.

    Returns:
        Tensor: Result.
    """
    if beta < 1e-5:
        return l1_loss(y_true, y_pred)

    error = subtract(y_true, y_pred)
    abs_error = abs(error)

    quadratic = divide(multiply(0.5, square(error)), beta)
    linear = subtract(abs_error, 0.5 * beta)

    loss = where(less_equal(abs_error, beta), quadratic, linear)
    return mean(loss)


def cosine_similarity_loss(y_true: Tensor, y_pred: Tensor, axis: int = -1) -> Any:  # type: ignore
    """Cosine Similarity Loss.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    y_true_norm = l2_normalize(y_true, axis=axis)
    y_pred_norm = l2_normalize(y_pred, axis=axis)

    cos_sim = sum(multiply(y_true_norm, y_pred_norm), axis=axis)
    # Loss is typically 1 - cosine_similarity
    return mean(subtract(1.0, cos_sim))


def kl_div_loss(y_true: Tensor, y_pred: Tensor) -> Any:  # type: ignore
    """Kullback-Leibler divergence loss.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.

    Returns:
        Tensor: Result.
    """
    # only compute where y_true > 0 to avoid log(0)
    safe_y_true = where(greater(y_true, 0.0), y_true, ones_like(y_true))
    true_log = multiply(y_true, log(safe_y_true))

    kl = subtract(true_log, multiply(y_true, y_pred))
    return mean(kl)


def hinge_loss(y_true: Tensor, y_pred: Tensor) -> Any:  # type: ignore
    """Hinge loss. y_true should be -1 or 1.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.

    Returns:
        Tensor: Result.
    """
    margin = subtract(1.0, multiply(y_true, y_pred))
    return mean(maximum(0.0, margin))


def gaussian_nll_loss(y_pred: Tensor, y_true: Tensor, var: Tensor, eps: float = 1e-6) -> Any:  # type: ignore
    """Gaussian Negative Log Likelihood loss.

    Args:
        y_pred (Tensor): The y_pred parameter.
        y_true (Tensor): The y_true parameter.
        var (Tensor): The var parameter.
        eps (float): The eps parameter.

    Returns:
        Tensor: Result.
    """
    # 0.5 * (log(max(var, eps)) + (y_true - y_pred)^2 / max(var, eps)) + const
    var_safe = maximum(var, eps)

    term1 = log(var_safe)
    term2 = divide(square(subtract(y_true, y_pred)), var_safe)

    loss = multiply(0.5, add(term1, term2))
    return mean(loss)


def log_cosh_loss(y_true: Tensor, y_pred: Tensor) -> Any:  # type: ignore
    """Log-Cosh loss.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.

    Returns:
        Tensor: Result.
    """
    # log(cosh(y_pred - y_true))
    # We can use the approximation: x + softplus(-2x) - log(2)
    # or just use log(cosh(x)) if cosh is safe.
    error = subtract(y_pred, y_true)

    # More numerically stable: log(cosh(x)) = |x| + softplus(-2|x|) - log(2)
    abs_error = abs(error)
    stable_logcosh = subtract(
        add(abs_error, softplus(multiply(-2.0, abs_error))),
        0.6931471805599453,  # log(2)
    )
    return mean(stable_logcosh)


def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0.0) -> Any:  # type: ignore
    """Margin Ranking Loss.

    Args:
        input1 (Tensor): The input1 parameter.
        input2 (Tensor): The input2 parameter.
        target (Tensor): The target parameter.
        margin (float): The margin parameter.

    Returns:
        Tensor: Result.
    """
    # max(0, -target * (input1 - input2) + margin)
    diff = subtract(input1, input2)
    loss = add(multiply(multiply(-1.0, target), diff), margin)
    return mean(maximum(0.0, loss))


def nll_loss(y_pred: Tensor, y_true: Tensor) -> Any:  # type: ignore
    """Negative log likelihood loss.

    Args:
        y_pred (Tensor): The y_pred parameter.
        y_true (Tensor): The y_true parameter.

    Returns:
        Tensor: Result.
    """
    # Gather the log probs corresponding to target indices
    # y_true needs to be expanded
    y_true_expanded = expand_dims(y_true, axis=-1)
    gathered = take_along_axis(y_pred, y_true_expanded, axis=-1)
    return mean(multiply(-1.0, gathered))


def triplet_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0, p: float = 2.0) -> Any:  # type: ignore
    """Triplet margin loss.

    Args:
        anchor (Tensor): The anchor parameter.
        positive (Tensor): The positive parameter.
        negative (Tensor): The negative parameter.
        margin (float): The margin parameter.
        p (float): The p parameter.

    Returns:
        Tensor: Result.
    """
    if p == 2.0:
        d_pos = sum(square(subtract(anchor, positive)), axis=-1)
        d_neg = sum(square(subtract(anchor, negative)), axis=-1)
    else:
        d_pos = sum(abs(subtract(anchor, positive)), axis=-1)
        d_neg = sum(abs(subtract(anchor, negative)), axis=-1)

    loss = maximum(0.0, add(subtract(d_pos, d_neg), margin))
    return mean(loss)


__all__ = [
    "cosine_similarity_loss",
    "gaussian_nll_loss",
    "hinge_loss",
    "huber_loss",
    "kl_div_loss",
    "kld",
    "kullback_leibler_divergence",
    "l1_loss",
    "log_cosh_loss",
    "logcosh",
    "mape",
    "mape_loss",
    "margin_ranking_loss",
    "mse_loss",
    "msle",
    "msle_loss",
    "nll_loss",
    "smooth_l1_loss",
    "triplet_loss",
    "ctc_loss",
    "circle_loss",
    "categorical_generalized_cross_entropy",
]


def msle_loss(y_true: Tensor, y_pred: Tensor) -> Any:  # type: ignore
    """Mean Squared Logarithmic Error.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.

    Returns:
        Tensor: Result.
    """
    # mean(square(log(y_pred + 1) - log(y_true + 1)))
    safe_pred = maximum(y_pred, 0.0)
    safe_true = maximum(y_true, 0.0)
    log_pred = log(add(safe_pred, 1.0))
    log_true = log(add(safe_true, 1.0))
    return mean(square(subtract(log_pred, log_true)))


def mape_loss(y_true: Tensor, y_pred: Tensor) -> Any:  # type: ignore
    """Mean Absolute Percentage Error.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.

    Returns:
        Tensor: Result.
    """
    # mean(abs((y_true - y_pred) / max(y_true, eps))) * 100
    eps = 1e-7
    diff = abs(subtract(y_true, y_pred))
    safe_true = maximum(abs(y_true), eps)
    return multiply(mean(divide(diff, safe_true)), 100.0)


# Aliases
msle = msle_loss
mape = mape_loss
kullback_leibler_divergence = kl_div_loss
kld = kl_div_loss
logcosh = log_cosh_loss

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("CircleLoss")
class CircleLoss(OpDef):
    """CircleLoss operation."""

    op_name = "CircleLoss"

    def infer_shape(self, *args, **kwargs):  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("CategoricalGeneralizedCrossEntropy")
class CategoricalGeneralizedCrossEntropy(OpDef):
    """CategoricalGeneralizedCrossEntropy operation."""

    op_name = "CategoricalGeneralizedCrossEntropy"

    def infer_shape(self, *args, **kwargs):  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


def ctc_loss(*args: Any, **kwargs: Any) -> Any:
    """ctc_loss function.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("CtcLoss")()(*args, **kwargs)


def circle_loss(*args: Any, **kwargs: Any) -> Any:
    """circle_loss function.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("CircleLoss")()(*args, **kwargs)


def categorical_generalized_cross_entropy(*args: Any, **kwargs: Any) -> Any:
    """categorical_generalized_cross_entropy function.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("CategoricalGeneralizedCrossEntropy")()(*args, **kwargs)
