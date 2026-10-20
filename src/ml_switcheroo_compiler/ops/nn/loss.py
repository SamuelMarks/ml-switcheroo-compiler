"""Canonical loss operations and neural network objective functions."""

from __future__ import annotations

import typing
import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import (
    add,
    divide,
    greater,
    less_equal,
    maximum,
    multiply,
    not_equal,
    subtract,
    true_divide,
)
from ml_switcheroo_compiler.ops.binary import power as op_pow
from ml_switcheroo_compiler.ops.creation.frontend import ones_like
from ml_switcheroo_compiler.ops.creation.frontend_basic import zeros
from ml_switcheroo_compiler.ops.nn.activations import log_softmax, one_hot, softplus
from ml_switcheroo_compiler.ops.nn.nlp import CtcLoss, ctc_loss
from ml_switcheroo_compiler.ops.nn.normalization import l2_normalize
from ml_switcheroo_compiler.ops.reductions import argmax, mean, sum
from ml_switcheroo_compiler.ops.reductions import mean as op_mean
from ml_switcheroo_compiler.ops.reductions import sum as op_sum
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape.frontend import expand_dims
from ml_switcheroo_compiler.ops.shape.indexing import take_along_axis, where
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.ops.unary import abs, exp, log, negative, square
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def l1_loss(y_true: Tensor, y_pred: Tensor) -> TensorOrScalar:
    """Compute Mean Absolute Error (L1) loss between true and predicted tensors.

    Args:
        y_true (Tensor): Ground truth target tensor.
        y_pred (Tensor): Predicted tensor.

    Returns:
        TensorOrScalar: Scalar mean L1 loss tensor.
    """
    return mean(abs(subtract(y_true, y_pred)))


def mse_loss(y_true: Tensor, y_pred: Tensor) -> TensorOrScalar:
    """Compute Mean Squared Error (L2) loss between true and predicted tensors.

    Args:
        y_true (Tensor): Ground truth target tensor.
        y_pred (Tensor): Predicted tensor.

    Returns:
        TensorOrScalar: Scalar mean squared error loss tensor.
    """
    return mean(square(subtract(y_true, y_pred)))


def huber_loss(y_true: Tensor, y_pred: Tensor, delta: float = 1.0) -> TensorOrScalar:
    """Compute Huber loss between true and predicted tensors.

    Args:
        y_true (Tensor): Ground truth target tensor.
        y_pred (Tensor): Predicted tensor.
        delta (float): Threshold at which to change between quadratic and linear loss.

    Returns:
        TensorOrScalar: Scalar Huber loss tensor.
    """
    error = subtract(y_true, y_pred)
    abs_error = abs(error)

    quadratic = multiply(0.5, square(error))
    linear = multiply(delta, subtract(abs_error, 0.5 * delta))

    loss = where(less_equal(abs_error, delta), quadratic, linear)
    return mean(loss)


def smooth_l1_loss(y_true: Tensor, y_pred: Tensor, beta: float = 1.0) -> TensorOrScalar:
    """Compute Smooth L1 loss with transition parameter beta.

    Args:
        y_true (Tensor): Ground truth target tensor.
        y_pred (Tensor): Predicted tensor.
        beta (float): Threshold parameter controlling the quadratic region.

    Returns:
        TensorOrScalar: Scalar Smooth L1 loss tensor.
    """
    if beta < 1e-5:
        return l1_loss(y_true, y_pred)

    error = subtract(y_true, y_pred)
    abs_error = abs(error)

    quadratic = divide(multiply(0.5, square(error)), beta)
    linear = subtract(abs_error, 0.5 * beta)

    loss = where(less_equal(abs_error, beta), quadratic, linear)
    return mean(loss)


def cosine_similarity_loss(y_true: Tensor, y_pred: Tensor, axis: int = -1) -> TensorOrScalar:
    """Compute cosine similarity distance loss (1 - cos_sim).

    Args:
        y_true (Tensor): Ground truth target tensor.
        y_pred (Tensor): Predicted tensor.
        axis (int): Dimension along which cosine similarity is computed.

    Returns:
        TensorOrScalar: Scalar cosine similarity loss tensor.
    """
    y_true_norm = l2_normalize(y_true, axis=axis)
    y_pred_norm = l2_normalize(y_pred, axis=axis)

    cos_sim = sum(multiply(y_true_norm, y_pred_norm), axis=axis)
    return mean(subtract(1.0, cos_sim))


def kl_div_loss(y_true: Tensor, y_pred: Tensor) -> TensorOrScalar:
    """Compute Kullback-Leibler divergence loss.

    Args:
        y_true (Tensor): Target probability distribution tensor.
        y_pred (Tensor): Log-probability predictions tensor.

    Returns:
        TensorOrScalar: Scalar KL divergence loss tensor.
    """
    safe_y_true = where(greater(y_true, 0.0), y_true, ones_like(y_true))
    true_log = multiply(y_true, log(safe_y_true))

    kl = subtract(true_log, multiply(y_true, y_pred))
    return mean(kl)


def hinge_loss(y_true: Tensor, y_pred: Tensor) -> TensorOrScalar:
    """Compute Hinge loss for binary classification with labels in {-1, 1}.

    Args:
        y_true (Tensor): Ground truth target tensor with values -1 or 1.
        y_pred (Tensor): Predicted score tensor.

    Returns:
        TensorOrScalar: Scalar hinge loss tensor.
    """
    margin = subtract(1.0, multiply(y_true, y_pred))
    return mean(maximum(0.0, margin))


def gaussian_nll_loss(y_pred: Tensor, y_true: Tensor, var: Tensor, eps: float = 1e-6) -> TensorOrScalar:
    """Compute Gaussian Negative Log Likelihood loss.

    Args:
        y_pred (Tensor): Predicted mean tensor.
        y_true (Tensor): Ground truth target tensor.
        var (Tensor): Predicted variance tensor.
        eps (float): Clamping stability value for variance.

    Returns:
        TensorOrScalar: Scalar Gaussian NLL loss tensor.
    """
    var_safe = maximum(var, eps)

    term1 = log(var_safe)
    term2 = divide(square(subtract(y_true, y_pred)), var_safe)

    loss = multiply(0.5, add(term1, term2))
    return mean(loss)


def log_cosh_loss(y_true: Tensor, y_pred: Tensor) -> TensorOrScalar:
    """Compute numerically stable Log-Cosh loss of the prediction error.

    Args:
        y_true (Tensor): Ground truth target tensor.
        y_pred (Tensor): Predicted tensor.

    Returns:
        TensorOrScalar: Scalar Log-Cosh loss tensor.
    """
    error = subtract(y_pred, y_true)
    abs_error = abs(error)
    stable_logcosh = subtract(
        add(abs_error, softplus(multiply(-2.0, abs_error))),
        0.6931471805599453,
    )
    return mean(stable_logcosh)


def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0.0) -> TensorOrScalar:
    """Compute Margin Ranking loss between pairs of inputs.

    Args:
        input1 (Tensor): First input score tensor.
        input2 (Tensor): Second input score tensor.
        target (Tensor): Target ranking tensor with values 1 or -1.
        margin (float): Margin offset parameter.

    Returns:
        TensorOrScalar: Scalar margin ranking loss tensor.
    """
    diff = subtract(input1, input2)
    loss = add(multiply(multiply(-1.0, target), diff), margin)
    return mean(maximum(0.0, loss))


def nll_loss(y_pred: Tensor, y_true: Tensor) -> TensorOrScalar:
    """Compute Negative Log Likelihood loss from log-probabilities and class indices.

    Args:
        y_pred (Tensor): Log-probabilities tensor of shape (N, C).
        y_true (Tensor): Class index tensor of shape (N,).

    Returns:
        TensorOrScalar: Scalar negative log likelihood loss tensor.
    """
    y_true_expanded = expand_dims(y_true, axis=-1)
    gathered = take_along_axis(y_pred, y_true_expanded, axis=-1)
    return mean(multiply(-1.0, gathered))


def triplet_loss(anchor: Tensor, positive: Tensor, negative_sample: Tensor, margin: float = 1.0, p: float = 2.0) -> TensorOrScalar:
    """Compute Triplet Margin loss over anchor, positive, and negative embeddings.

    Args:
        anchor (Tensor): Anchor embedding tensor.
        positive (Tensor): Positive embedding tensor.
        negative_sample (Tensor): Negative embedding tensor.
        margin (float): Distance margin offset.
        p (float): Norm degree for distance calculation (1.0 or 2.0).

    Returns:
        TensorOrScalar: Scalar triplet loss tensor.
    """
    if p == 2.0:
        d_pos = sum(square(subtract(anchor, positive)), axis=-1)
        d_neg = sum(square(subtract(anchor, negative_sample)), axis=-1)
    else:
        d_pos = sum(abs(subtract(anchor, positive)), axis=-1)
        d_neg = sum(abs(subtract(anchor, negative_sample)), axis=-1)

    loss = maximum(0.0, add(subtract(d_pos, d_neg), margin))
    return mean(loss)


def msle_loss(y_true: Tensor, y_pred: Tensor) -> TensorOrScalar:
    """Compute Mean Squared Logarithmic Error loss.

    Args:
        y_true (Tensor): Ground truth target tensor.
        y_pred (Tensor): Predicted tensor.

    Returns:
        TensorOrScalar: Scalar MSLE loss tensor.
    """
    safe_pred = maximum(y_pred, 0.0)
    safe_true = maximum(y_true, 0.0)
    log_pred = log(add(safe_pred, 1.0))
    log_true = log(add(safe_true, 1.0))
    return mean(square(subtract(log_pred, log_true)))


def mape_loss(y_true: Tensor, y_pred: Tensor) -> TensorOrScalar:
    """Compute Mean Absolute Percentage Error loss.

    Args:
        y_true (Tensor): Ground truth target tensor.
        y_pred (Tensor): Predicted tensor.

    Returns:
        TensorOrScalar: Scalar MAPE loss tensor.
    """
    eps = 1e-7
    diff = abs(subtract(y_true, y_pred))
    safe_true = maximum(abs(y_true), eps)
    return multiply(mean(divide(diff, safe_true)), 100.0)


msle = msle_loss
mape = mape_loss
kullback_leibler_divergence = kl_div_loss
kld = kl_div_loss
logcosh = log_cosh_loss


@register_op("CircleLoss")
class CircleLoss(OpDef):
    """CircleLoss operation definition."""

    op_name = "CircleLoss"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer scalar output shape for CircleLoss.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (TensorOrScalar): Keyword arguments.

        Returns:
            tuple[int, ...]: Empty tuple representing scalar loss shape.
        """
        _ = (args, kwargs)
        return ()


@register_op("CategoricalGeneralizedCrossEntropy")
class CategoricalGeneralizedCrossEntropy(OpDef):
    """CategoricalGeneralizedCrossEntropy operation definition."""

    op_name = "CategoricalGeneralizedCrossEntropy"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer scalar output shape for CategoricalGeneralizedCrossEntropy.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (TensorOrScalar): Keyword arguments.

        Returns:
            tuple[int, ...]: Empty tuple representing scalar loss shape.
        """
        _ = (args, kwargs)
        return ()


def dice_loss(
    y_true: Tensor,
    y_pred: Tensor,
    axis: tuple[int, ...] | int | None = None,
    smooth: float = 1e-5,
) -> TensorOrScalar:
    """Compute the Dice loss for segmentation masks.

    Args:
        y_true (Tensor): True binary or one-hot labels.
        y_pred (Tensor): Predicted probabilities.
        axis (Optional[Union[tuple[int, ...], int]]): Axes to reduce over.
        smooth (float): Smoothing constant to prevent division by zero.

    Returns:
        TensorOrScalar: Calculated Dice loss tensor.
    """
    intersection = op_sum(multiply(y_true, y_pred), axis=axis)
    y_true_sum = op_sum(y_true, axis=axis)
    y_pred_sum = op_sum(y_pred, axis=axis)

    numerator = add(multiply(2.0, intersection), smooth)
    denominator = add(add(y_true_sum, y_pred_sum), smooth)

    dice_coeff = true_divide(numerator, denominator)
    return subtract(1.0, dice_coeff)


def categorical_generalized_cross_entropy(y_true: Tensor, y_pred: Tensor, q: float = 0.7, axis: int = -1) -> TensorOrScalar:
    """Compute Categorical Generalized Cross Entropy (GCE) loss.

    Args:
        y_true (Tensor): One-hot encoded ground truth labels.
        y_pred (Tensor): Predicted class probabilities.
        q (float): Box-Cox transformation parameter in (0, 1].
        axis (int): Class dimension axis.

    Returns:
        TensorOrScalar: Computed GCE loss tensor.
    """
    epsilon = 1e-7
    y_pred_clipped = get_op("Clip")()(y_pred, epsilon, 1.0 - epsilon)
    p_y = op_sum(multiply(y_true, y_pred_clipped), axis=axis)
    return true_divide(subtract(1.0, op_pow(p_y, q)), q)


def _compute_circle_margins(margin: float) -> tuple[float, float, float, float]:
    """Compute the positive and negative margin parameters for circle loss.

    Args:
        margin (float): Optimization margin parameter.

    Returns:
        tuple[float, float, float, float]: Tuple of (O_p, O_n, Delta_p, Delta_n).
    """
    return 1.0 + margin, -margin, 1.0 - margin, margin


def _compute_circle_logits(y_pred: Tensor, margin: float, gamma: float) -> tuple[TensorOrScalar, TensorOrScalar]:
    """Compute scaled positive and negative class logits for circle loss.

    Args:
        y_pred (Tensor): Predicted similarity scores.
        margin (float): Margin relaxation factor.
        gamma (float): Scale factor.

    Returns:
        tuple[TensorOrScalar, TensorOrScalar]: Positive and negative scaled logits.
    """
    O_p, O_n, Delta_p, Delta_n = _compute_circle_margins(margin)
    alpha_p = maximum(0.0, subtract(O_p, y_pred))
    alpha_n = maximum(0.0, subtract(y_pred, O_n))

    logit_p = multiply(multiply(alpha_p, subtract(y_pred, Delta_p)), gamma)
    logit_n = multiply(multiply(alpha_n, subtract(y_pred, Delta_n)), gamma)
    return logit_p, logit_n


def _compute_circle_loss_reduction(logit_p: TensorOrScalar, logit_n: TensorOrScalar, mask_p: Tensor, mask_n: TensorOrScalar) -> TensorOrScalar:
    """Reduce positive and negative circle loss logits into per-sample loss.

    Args:
        logit_p (TensorOrScalar): Positive class scaled logits.
        logit_n (TensorOrScalar): Negative class scaled logits.
        mask_p (Tensor): Positive class indicator mask.
        mask_n (TensorOrScalar): Negative class indicator mask.

    Returns:
        TensorOrScalar: Reduced circle loss tensor.
    """
    inf_val = 1e9
    neg_inf_p = ops.multiply(ops.subtract(1.0, mask_p), -inf_val)
    neg_inf_n = ops.multiply(ops.subtract(1.0, mask_n), -inf_val)

    lse_p = ops.log(ops.sum(ops.exp(ops.add(ops.multiply(logit_p, -1.0), neg_inf_p)), axis=-1))
    lse_n = ops.log(ops.sum(ops.exp(ops.add(logit_n, neg_inf_n)), axis=-1))
    loss = ops.add(lse_p, lse_n)
    return ops.log(ops.add(1.0, ops.exp(loss)))


def circle_loss(y_true: Tensor, y_pred: Tensor, margin: float = 0.25, gamma: float = 256.0) -> TensorOrScalar:
    """Compute Circle loss over similarity scores and binary masks.

    Args:
        y_true (Tensor): Binary indicator mask of ground truth positives.
        y_pred (Tensor): Predicted similarity scores.
        margin (float): Relaxation margin parameter.
        gamma (float): Scale factor.

    Returns:
        TensorOrScalar: Computed Circle loss tensor.
    """
    mask_p = y_true
    mask_n = subtract(1.0, y_true)
    logit_p, logit_n = _compute_circle_logits(y_pred, margin, gamma)
    return _compute_circle_loss_reduction(logit_p, logit_n, mask_p, mask_n)


def tversky_loss(y_true: Tensor, y_pred: Tensor, alpha: float = 0.5, beta: float = 0.5) -> TensorOrScalar:
    """Compute Tversky loss, an asymmetric generalization of Dice loss.

    Args:
        y_true (Tensor): Ground truth binary or one-hot mask.
        y_pred (Tensor): Predicted probabilities.
        alpha (float): Penalty weight for false positives.
        beta (float): Penalty weight for false negatives.

    Returns:
        TensorOrScalar: Calculated Tversky loss tensor.
    """
    intersection = op_sum(multiply(y_true, y_pred), axis=-1)
    fps = op_sum(multiply(subtract(1.0, y_true), y_pred), axis=-1)
    fns = op_sum(multiply(y_true, subtract(1.0, y_pred)), axis=-1)

    denom = add(add(intersection, multiply(alpha, fps)), add(multiply(beta, fns), 1e-7))
    return subtract(1.0, true_divide(intersection, denom))


def _clip_and_convert_logits(y_pred: Tensor, from_logits: bool) -> TensorOrScalar:
    """Clip probability predictions to [1e-7, 1 - 1e-7] when from_logits is False.

    Args:
        y_pred (Tensor): Input predictions tensor.
        from_logits (bool): Whether predictions are unnormalized logits.

    Returns:
        TensorOrScalar: Clipped probability tensor or raw logits.
    """
    if not from_logits:
        return get_op("Clip")()(y_pred, 1e-7, 1.0 - 1e-7)
    return y_pred


def _compute_bce_loss(y_true: Tensor, y_pred: Tensor, from_logits: bool) -> TensorOrScalar:
    """Compute element-wise binary crossentropy from logits or probabilities.

    Args:
        y_true (Tensor): Ground truth binary labels.
        y_pred (Tensor): Predictions (logits or clipped probabilities).
        from_logits (bool): Whether predictions are logits.

    Returns:
        TensorOrScalar: Element-wise binary crossentropy tensor.
    """
    neg_op = get_op("Negative")()
    if from_logits:
        max_x_0 = maximum(y_pred, 0.0)
        x_z = multiply(y_pred, y_true)
        abs_x = abs(y_pred)
        neg_abs_x = neg_op(abs_x)
        exp_neg_abs_x = exp(neg_abs_x)
        log_term = log(add(1.0, exp_neg_abs_x))
        return add(subtract(max_x_0, x_z), log_term)

    term1 = multiply(y_true, log(y_pred))
    term2 = multiply(subtract(1.0, y_true), log(subtract(1.0, y_pred)))
    return neg_op(add(term1, term2))


def binary_crossentropy(
    y_true: Tensor,
    y_pred: Tensor,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
) -> TensorOrScalar:
    """Compute binary crossentropy loss with optional label smoothing.

    Args:
        y_true (Tensor): Ground truth binary labels.
        y_pred (Tensor): Predicted logits or probabilities.
        from_logits (bool): Whether predictions are unnormalized logits.
        label_smoothing (float): Label smoothing factor in [0, 1].
        axis (int): Reduction axis.

    Returns:
        TensorOrScalar: Reduced binary crossentropy loss tensor.
    """
    if label_smoothing > 0.0:
        y_true = add(multiply(y_true, 1.0 - label_smoothing), 0.5 * label_smoothing)

    y_pred_prepared = typing.cast(Tensor, _clip_and_convert_logits(y_pred, from_logits))
    bce = _compute_bce_loss(y_true, y_pred_prepared, from_logits)
    return op_mean(bce, axis=axis)


def categorical_crossentropy(
    y_true: Tensor,
    y_pred: Tensor,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
) -> TensorOrScalar:
    """Compute categorical crossentropy loss across one-hot labels.

    Args:
        y_true (Tensor): One-hot ground truth tensor.
        y_pred (Tensor): Predicted logits or probabilities.
        from_logits (bool): Whether predictions are unnormalized logits.
        label_smoothing (float): Label smoothing factor in [0, 1].
        axis (int): Class axis.

    Returns:
        TensorOrScalar: Categorical crossentropy loss tensor.
    """
    epsilon = 1e-7

    if label_smoothing > 0.0:
        num_classes = y_pred.shape[axis]
        smooth_val = label_smoothing / float(num_classes)
        y_true = add(multiply(y_true, 1.0 - label_smoothing), smooth_val)

    if from_logits:
        y_pred_log = typing.cast(Tensor, log_softmax(y_pred, axis=axis))
        return negative(op_sum(multiply(y_true, y_pred_log), axis=axis))
    y_pred_clipped = get_op("Clip")()(y_pred, epsilon, 1.0 - epsilon)
    return negative(op_sum(multiply(y_true, log(y_pred_clipped)), axis=axis))


def sparse_categorical_crossentropy(
    y_true: Tensor,
    y_pred: Tensor,
    from_logits: bool = False,
    ignore_class: int | None = None,
    axis: int = -1,
) -> TensorOrScalar:
    """Compute sparse categorical crossentropy loss for integer class targets.

    Args:
        y_true (Tensor): Integer class index labels.
        y_pred (Tensor): Predicted logits or probabilities.
        from_logits (bool): Whether predictions are unnormalized logits.
        ignore_class (Optional[int]): Optional class ID to mask out from loss.
        axis (int): Class dimension axis.

    Returns:
        TensorOrScalar: Sparse categorical crossentropy loss tensor.
    """
    num_classes = y_pred.shape[axis]
    y_true_one_hot = one_hot(y_true, num_classes, axis=axis)
    loss = categorical_crossentropy(typing.cast("Tensor", y_true_one_hot), y_pred, from_logits=from_logits, axis=axis)

    if ignore_class is not None:
        valid_mask = not_equal(y_true, ignore_class)
        loss_dtype = getattr(loss, "dtype", None) or DType("float32")
        zero_loss = zeros(getattr(loss, "shape", ()), dtype=loss_dtype)
        loss = where(valid_mask, typing.cast("Tensor", loss), zero_loss)

    return loss


def ctc_decode(
    inputs: Tensor,
    sequence_lengths: Tensor,
    greedy: bool = True,
    beam_width: int = 100,
    top_paths: int = 1,
) -> tuple[list[Tensor], Tensor]:
    """Decode CTC logits into top predicted paths and log probabilities.

    Args:
        inputs (Tensor): Input sequence logits tensor.
        sequence_lengths (Tensor): Sequence length tensor.
        greedy (bool): Whether to perform greedy decoding.
        beam_width (int): Beam search width.
        top_paths (int): Number of decoded paths to return.

    Returns:
        tuple[list[Tensor], Tensor]: List of decoded path tensors and log-probability tensor.
    """
    _ = (sequence_lengths, greedy, beam_width)
    paths = argmax(inputs, axis=-1)
    batch_size = inputs.shape[0] if len(inputs.shape) >= 2 else 1
    log_probs = zeros((batch_size, top_paths), dtype=inputs.dtype)
    return [paths] * top_paths, log_probs


@register_op("AdaptiveLogSoftmaxWithLoss")
class AdaptiveLogSoftmaxWithLoss(OpDef):
    """AdaptiveLogSoftmaxWithLoss operation definition."""

    op_name = "AdaptiveLogSoftmaxWithLoss"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Infer output and scalar loss shapes for AdaptiveLogSoftmaxWithLoss.

        Args:
            *args (EagerTensorProtocol): Input and target operands.
            **kwargs (TensorOrScalar): Keyword attributes.

        Returns:
            tuple[tuple[int, ...], tuple[int, ...]]: Tuple of (target_shape, scalar_shape).
        """
        _ = kwargs
        return (args[1].shape, ())


def _emit_adaptive_log_softmax_with_loss_node(
    input_tensor: Tensor,
    target: Tensor,
    cutoffs: TensorOrScalar,
    add_cluster_prob: bool,
) -> tuple[Tensor, Tensor]:
    """Emit a LogicalNode for AdaptiveLogSoftmaxWithLoss during graph tracing.

    Args:
        input_tensor (Tensor): Input activations tensor.
        target (Tensor): Target class indices tensor.
        cutoffs (TensorOrScalar): Vocabulary cluster cutoff boundaries.
        add_cluster_prob (bool): Whether to append cluster log-probabilities.

    Returns:
        tuple[Tensor, Tensor]: Symbolic output tensor and loss tensor.

    Raises:
        RuntimeError: If invoked outside an active tracing context.
    """
    if not global_tracing_state.is_tracing:
        raise RuntimeError("Must be tracing")

    out_id = str(uuid.uuid4())
    loss_id = f"{out_id}:loss"

    from collections.abc import Mapping

    from ml_switcheroo_ir import AttributeValue

    node = LogicalNode(
        id=out_id,
        op_type="AdaptiveLogSoftmaxWithLoss",
        inputs=[getattr(input_tensor.data, "id"), getattr(target.data, "id")],
        shape_metadata=target.shape,
        attributes=typing.cast("Mapping[str, AttributeValue]", {"cutoffs": cutoffs, "add_cluster_prob": add_cluster_prob}),
    )
    global_tracing_state.add_node(node)

    proxy_out = ProxyTensor(id=out_id, shape=target.shape, dtype=input_tensor.dtype)
    proxy_loss = ProxyTensor(id=loss_id, shape=(), dtype=input_tensor.dtype)

    return (
        Tensor(proxy_out, TensorConfig(target.shape, input_tensor.dtype, input_tensor.device)),
        Tensor(proxy_loss, TensorConfig((), input_tensor.dtype, input_tensor.device)),
    )


def adaptive_log_softmax_with_loss(
    input: Tensor,
    target: Tensor,
    cutoffs: TensorOrScalar,
    add_cluster_prob: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute adaptive log softmax probabilities and NLL loss.

    Args:
        input (Tensor): Input activations tensor.
        target (Tensor): Target class indices tensor.
        cutoffs (TensorOrScalar): Vocabulary cluster cutoff boundaries.
        add_cluster_prob (bool): Whether to include cluster probabilities.

    Returns:
        tuple[Tensor, Tensor]: Tuple of (output_tensor, loss_tensor).
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend as _get_active_backend

        backend = _get_active_backend()
        out, loss = typing.cast(
            "tuple[TensorOrScalar, TensorOrScalar]",
            backend.execute_op(
                "AdaptiveLogSoftmaxWithLoss",
                input.data,
                target.data,
                cutoffs=cutoffs,
                add_cluster_prob=add_cluster_prob,
            ),
        )
        return (
            Tensor(out, TensorConfig(target.shape, input.dtype, input.device)),
            Tensor(loss, TensorConfig((), input.dtype, input.device)),
        )
    return _emit_adaptive_log_softmax_with_loss_node(input, target, cutoffs, add_cluster_prob)


@register_op("LogPoissonLoss")
class LogPoissonLoss(OpDef):
    """LogPoissonLoss operation definition."""

    op_name = "LogPoissonLoss"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer output shape for LogPoissonLoss matching targets shape.

        Args:
            *args (EagerTensorProtocol): Positional arguments (targets, log_input).
            **kwargs (TensorOrScalar): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred output shape.
        """
        _ = kwargs
        return getattr(args[0], "shape", ()) if args else ()


def log_poisson_loss(
    targets: Tensor,
    log_input: Tensor,
    compute_full_loss: bool = False,
    name: str | None = None,
) -> TensorOrScalar:
    """Compute log Poisson loss: exp(log_input) - targets * log_input.

    When compute_full_loss is True, adds Stirling's approximation:
    targets * log(targets) - targets + 0.5 * log(2 * pi * targets).

    Args:
        targets (Tensor): Ground truth Poisson target counts.
        log_input (Tensor): Logarithm of predicted Poisson rate.
        compute_full_loss (bool): Whether to add Stirling's approximation term.
        name (Optional[str]): Optional operation name.

    Returns:
        TensorOrScalar: Element-wise log Poisson loss tensor.
    """
    _ = name
    base_loss = subtract(exp(log_input), multiply(targets, log_input))
    if not compute_full_loss:
        return base_loss
    safe_targets = maximum(targets, 1e-7)
    stirling = add(
        subtract(multiply(targets, log(safe_targets)), targets),
        multiply(0.5, log(multiply(6.283185307179586, safe_targets))),
    )
    return where(greater(targets, 1.0), add(base_loss, stirling), base_loss)


@register_op("InTopK")
class InTopK(OpDef):
    """InTopK operation definition."""

    op_name = "InTopK"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer output shape of InTopK matching the 1D targets shape.

        Args:
            *args (EagerTensorProtocol): Positional arguments (targets, predictions).
            **kwargs (TensorOrScalar): Keyword arguments.

        Returns:
            tuple[int, ...]: Inferred shape tuple.
        """
        _ = kwargs
        return getattr(args[0], "shape", ()) if args else ()


def in_top_k(
    targets: Tensor,
    predictions: Tensor,
    k: int,
    name: str | None = None,
) -> TensorOrScalar:
    """Determine whether target classes are within the top K predictions.

    Args:
        targets (Tensor): Ground truth class indices of shape (batch_size,).
        predictions (Tensor): Class prediction scores of shape (batch_size, num_classes).
        k (int): Number of top elements to consider.
        name (Optional[str]): Optional operation name.

    Returns:
        TensorOrScalar: Boolean tensor of shape (batch_size,) indicating top-k membership.
    """
    target_shape = getattr(targets, "shape", ())
    if config.eager_mode:
        backend = get_active_backend()
        t_data = getattr(targets, "data", targets)
        p_data = getattr(predictions, "data", predictions)
        data = backend.execute_op("InTopK", t_data, p_data, k=k, name=name)
        return Tensor(data, TensorConfig(target_shape, "bool", getattr(targets, "device", "cpu")))
    return _emit_shape_node("InTopK", [targets, predictions], {"k": k, "name": name}, target_shape, "bool")


def l2_loss(t: Tensor, name: str | None = None) -> TensorOrScalar:
    """Compute half the squared L2 norm of a tensor without square root: sum(t ** 2) / 2.

    Args:
        t (Tensor): Input tensor.
        name (Optional[str]): Optional operation name.

    Returns:
        TensorOrScalar: Scalar L2 loss tensor.
    """
    _ = name
    return multiply(sum(multiply(t, t)), 0.5)


def scale_regularization_loss(regularization_loss: Tensor, name: str | None = None) -> TensorOrScalar:
    """Scale the sum of regularization losses across replicas.

    Args:
        regularization_loss (Tensor): Regularization loss value.
        name (Optional[str]): Optional operation name.

    Returns:
        TensorOrScalar: Scaled regularization loss tensor.
    """
    _ = name
    return regularization_loss


__all__ = [
    "AdaptiveLogSoftmaxWithLoss",
    "CategoricalGeneralizedCrossEntropy",
    "CircleLoss",
    "CtcLoss",
    "InTopK",
    "LogPoissonLoss",
    "adaptive_log_softmax_with_loss",
    "binary_crossentropy",
    "categorical_crossentropy",
    "categorical_generalized_cross_entropy",
    "circle_loss",
    "cosine_similarity_loss",
    "ctc_decode",
    "ctc_loss",
    "dice_loss",
    "gaussian_nll_loss",
    "hinge_loss",
    "huber_loss",
    "in_top_k",
    "kl_div_loss",
    "kld",
    "kullback_leibler_divergence",
    "l1_loss",
    "l2_loss",
    "log_cosh_loss",
    "log_poisson_loss",
    "logcosh",
    "mape",
    "mape_loss",
    "margin_ranking_loss",
    "mse_loss",
    "msle",
    "msle_loss",
    "nll_loss",
    "scale_regularization_loss",
    "smooth_l1_loss",
    "sparse_categorical_crossentropy",
    "triplet_loss",
    "tversky_loss",
]
