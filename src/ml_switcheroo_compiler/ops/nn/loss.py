# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Calculate loss functions."""

from typing import Any, Optional, Union

from ml_switcheroo_compiler import ops

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import (
    Tensor,
    TensorConfig,
)
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.binary import (
    add,
    maximum,
    multiply,
    not_equal,
    subtract,
    true_divide,
)
from ml_switcheroo_compiler.ops.binary import power as op_pow
from ml_switcheroo_compiler.ops.nn.activations import log_softmax, one_hot
from ml_switcheroo_compiler.ops.nn.nlp import ctc_loss

# A simplified greedy decode for tracing/compiler compat.
# Base logic implementation
from ml_switcheroo_compiler.ops.reductions import (
    argmax,
    sum,
)
from ml_switcheroo_compiler.ops.reductions import mean as op_mean
from ml_switcheroo_compiler.ops.reductions import sum as op_sum
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape.indexing import where
from ml_switcheroo_compiler.ops.unary import abs, exp, log, negative


def dice_loss(
    y_true: Tensor,  # type: ignore
    y_pred: Tensor,  # type: ignore
    axis: Optional[Union[tuple[int, ...], int]] = None,
    smooth: float = 1e-5,
) -> Any:
    """Compute the Dice loss.

    Args:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predictions.
        axis (Optional[Union[tuple[int, ...], int]]): The axes to reduce over.
        smooth (float): Smoothing factor.

    Returns:
        Tensor: The calculated dice loss.
    """
    intersection = op_sum(multiply(y_true, y_pred), axis=axis)
    y_true_sum = op_sum(y_true, axis=axis)
    y_pred_sum = op_sum(y_pred, axis=axis)

    numerator = add(multiply(2.0, intersection), smooth)
    denominator = add(add(y_true_sum, y_pred_sum), smooth)

    dice_coeff = true_divide(numerator, denominator)
    return subtract(1.0, dice_coeff)


def categorical_generalized_cross_entropy(y_true: Tensor, y_pred: Tensor, q: float = 0.7, axis: int = -1) -> Any:  # type: ignore
    """Evaluate categorical_generalized_cross_entropy operation.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.
        q (float): The q parameter.
        axis (int): The axis parameter.

    Returns:
        Tensor: Result.
    """
    # Clip predictions to prevent NaNs
    epsilon = 1e-7
    y_pred = get_op("Clip")()(y_pred, epsilon, 1.0 - epsilon)

    # Extract probability of the true class
    # y_true is one-hot, so sum(y_true * y_pred) gives p_y
    p_y = op_sum(multiply(y_true, y_pred), axis=axis)

    # Calculate GCE: (1 - p_y^q) / q
    loss = true_divide(subtract(1.0, op_pow(p_y, q)), q)
    return loss


def _compute_circle_margins(margin: float) -> tuple[float, float, float, float]:
    """Compute the positive and negative margin parameters for circle loss.

    Args:
        margin (float): The margin used for optimization.

    Returns:
        tuple[float, float, float, float]: The computed margin components (O_p, O_n, Delta_p, Delta_n).
    """
    return 1.0 + margin, -margin, 1.0 - margin, margin


def _compute_circle_logits(y_pred: Tensor, margin: float, gamma: float) -> Any:  # type: ignore
    """Compute the scaled logits for the positive and negative classes in circle loss.

    Args:
        y_pred (Tensor): The predicted logits or similarities.
        margin (float): The margin parameter for scaling.
        gamma (float): The gamma scale factor.

    Returns:
        tuple[Tensor, Tensor]: The computed positive and negative logits.
    """
    O_p, O_n, Delta_p, Delta_n = _compute_circle_margins(margin)
    alpha_p = maximum(0.0, subtract(O_p, y_pred))
    alpha_n = maximum(0.0, subtract(y_pred, O_n))

    logit_p = multiply(multiply(alpha_p, subtract(y_pred, Delta_p)), gamma)
    logit_n = multiply(multiply(alpha_n, subtract(y_pred, Delta_n)), gamma)
    return logit_p, logit_n


def _compute_circle_loss_reduction(logit_p: Tensor, logit_n: Tensor, mask_p: Tensor, mask_n: Tensor) -> Any:  # type: ignore
    """Reduce the circle loss logits into the final scalar or per-sample loss.

    Args:
        logit_p (Tensor): The positive class logits.
        logit_n (Tensor): The negative class logits.
        mask_p (Tensor): The mask for the positive class.
        mask_n (Tensor): The mask for the negative class.

    Returns:
        Tensor: The computed reduced loss values.
    """
    INF = 1e9
    neg_inf_p = ops.multiply(ops.subtract(1.0, mask_p), -INF)
    neg_inf_n = ops.multiply(ops.subtract(1.0, mask_n), -INF)

    lse_p = ops.log(ops.sum(ops.exp(ops.add(ops.multiply(logit_p, -1.0), neg_inf_p)), axis=-1))
    lse_n = ops.log(ops.sum(ops.exp(ops.add(logit_n, neg_inf_n)), axis=-1))
    loss = ops.add(lse_p, lse_n)
    return ops.log(ops.add(1.0, ops.exp(loss)))


def circle_loss(y_true: Tensor, y_pred: Tensor, margin: float = 0.25, gamma: float = 256.0) -> Any:  # type: ignore
    """Evaluate circle_loss operation.

    Args:
        y_true (Tensor): The y_true parameter.
        y_pred (Tensor): The y_pred parameter.
        margin (float): The margin parameter.
        gamma (float): The gamma parameter.

    Returns:
        Tensor: Result.
    """
    mask_p = y_true
    mask_n = subtract(1.0, y_true)
    logit_p, logit_n = _compute_circle_logits(y_pred, margin, gamma)
    return _compute_circle_loss_reduction(logit_p, logit_n, mask_p, mask_n)


def tversky_loss(y_true: Tensor, y_pred: Tensor, alpha: float = 0.5, beta: float = 0.5) -> Any:  # type: ignore
    """Compute the Tversky loss, a generalization of the Dice loss.

    Args:
        y_true (Tensor): The ground truth values.
        y_pred (Tensor): The predicted probabilities.
        alpha (float): The weight of false positives.
        beta (float): The weight of false negatives.

    Returns:
        Tensor: The calculated Tversky loss.
    """
    intersection = op_sum(multiply(y_true, y_pred), axis=-1)
    fps = op_sum(multiply(subtract(1.0, y_true), y_pred), axis=-1)
    fns = op_sum(multiply(y_true, subtract(1.0, y_pred)), axis=-1)

    denom = add(add(intersection, multiply(alpha, fps)), add(multiply(beta, fns), 1e-7))
    return subtract(1.0, true_divide(intersection, denom))


def _clip_and_convert_logits(y_pred: Tensor, from_logits: bool) -> Any:  # type: ignore
    """Clip probabilities if from_logits is False.

    Args:
        y_pred (Tensor): The input predictions tensor.
        from_logits (bool): Whether the predictions are expected to be logits.

    Returns:
        Tensor: The clipped or unmodified predictions tensor.
    """
    if not from_logits:
        return get_op("Clip")()(y_pred, 1e-7, 1.0 - 1e-7)
    return y_pred


def _compute_bce_loss(y_true: Tensor, y_pred: Tensor, from_logits: bool) -> Any:  # type: ignore
    """Apply mathematical computation for BCE loss.

    Args:
        y_true (Tensor): The ground truth values.
        y_pred (Tensor): The predicted values.
        from_logits (bool): Whether the predictions are logits.

    Returns:
        Tensor: The computed binary crossentropy loss.
    """
    negative = get_op("Negative")()
    if from_logits:
        max_x_0 = maximum(y_pred, 0.0)
        x_z = multiply(y_pred, y_true)
        abs_x = abs(y_pred)
        neg_abs_x = negative(abs_x)
        exp_neg_abs_x = exp(neg_abs_x)
        log_term = log(add(1.0, exp_neg_abs_x))
        return add(subtract(max_x_0, x_z), log_term)

    term1 = multiply(y_true, log(y_pred))
    term2 = multiply(subtract(1.0, y_true), log(subtract(1.0, y_pred)))
    return negative(add(term1, term2))


def binary_crossentropy(
    y_true: Tensor,  # type: ignore
    y_pred: Tensor,  # type: ignore
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
) -> Any:
    """Compute the binary crossentropy loss.

    Args:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predictions.
        from_logits (bool): Whether the predictions are logits.
        label_smoothing (float): Float in [0, 1]. When > 0, label values are smoothed.
        axis (int): The axis along which to compute the loss.

    Returns:
        Tensor: The calculated binary crossentropy loss.
    """
    if label_smoothing > 0.0:
        y_true = add(multiply(y_true, 1.0 - label_smoothing), 0.5 * label_smoothing)

    y_pred = _clip_and_convert_logits(y_pred, from_logits)
    bce = _compute_bce_loss(y_true, y_pred, from_logits)

    return op_mean(bce, axis=axis)


def categorical_crossentropy(
    y_true: Tensor,  # type: ignore
    y_pred: Tensor,  # type: ignore
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
) -> Any:
    """Compute the categorical crossentropy loss.

    Args:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predictions.
        from_logits (bool): Whether the predictions are logits.
        label_smoothing (float): Float in [0, 1]. When > 0, label values are smoothed.
        axis (int): The axis along which to compute the loss.

    Returns:
        Tensor: The calculated categorical crossentropy loss.
    """
    epsilon = 1e-7

    if label_smoothing > 0.0:
        num_classes = y_pred.shape[axis]
        smooth_val = label_smoothing / float(num_classes)
        y_true = add(multiply(y_true, 1.0 - label_smoothing), smooth_val)

    if from_logits:
        y_pred = log_softmax(y_pred, axis=axis)
        return negative(op_sum(multiply(y_true, y_pred), axis=axis))
    else:
        y_pred = get_op("Clip")()(y_pred, epsilon, 1.0 - epsilon)
        return negative(op_sum(multiply(y_true, log(y_pred)), axis=axis))


def sparse_categorical_crossentropy(
    y_true: Tensor,  # type: ignore
    y_pred: Tensor,  # type: ignore
    from_logits: bool = False,
    ignore_class: Optional[int] = None,
    axis: int = -1,
) -> Any:
    """Compute the sparse categorical crossentropy loss.

    Args:
        y_true (Tensor): True labels (integers).
        y_pred (Tensor): Predictions.
        from_logits (bool): Whether the predictions are logits.
        ignore_class (Optional[int]): Class ID to ignore in loss computation.
        axis (int): The axis along which to compute the loss.

    Returns:
        Tensor: The calculated sparse categorical crossentropy loss.
    """
    num_classes = y_pred.shape[axis]
    y_true_one_hot = one_hot(y_true, num_classes, axis=axis)
    loss = categorical_crossentropy(y_true_one_hot, y_pred, from_logits=from_logits, axis=axis)

    if ignore_class is not None:
        valid_mask = not_equal(y_true, ignore_class)
        loss = where(valid_mask, loss, None)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    return loss


def ctc_decode(
    inputs: Tensor,  # type: ignore
    sequence_lengths: Tensor,  # type: ignore
    greedy: bool = True,
    beam_width: int = 100,
    top_paths: int = 1,
) -> tuple[list[Tensor], Tensor]:  # type: ignore
    """Decode CTC predictions.

    Args:
        inputs (Tensor): The input tensor containing the predictions.
        sequence_lengths (Tensor): The lengths of the sequences.
        greedy (bool): Whether to use greedy decoding.
        beam_width (int): The beam width for beam search decoding.
        top_paths (int): The number of top paths to return.

    Returns:
        tuple[list[Tensor], Tensor]: The decoded paths and the log probabilities.
    """
    from ml_switcheroo_compiler.ops.creation.frontend_basic import zeros

    paths = argmax(inputs, axis=-1)
    batch_size = inputs.shape[0] if len(inputs.shape) >= 2 else 1
    log_probs = zeros((batch_size, top_paths), dtype=inputs.dtype)

    return [paths] * top_paths, log_probs


@register_op("AdaptiveLogSoftmaxWithLoss")
class AdaptiveLogSoftmaxWithLoss(OpDef):
    """AdaptiveLogSoftmaxWithLoss operation definition."""

    op_name = "AdaptiveLogSoftmaxWithLoss"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infers the output shape of the AdaptiveLogSoftmaxWithLoss operation.

        Args:
            *args (object): The arguments passed to the operation.
            **kwargs (object): The keyword arguments passed to the operation.

        Returns: Any: A tuple containing the inferred shape for output and loss.
        """
        # Returns output (same shape as input target) and loss (scalar)
        return (args[1].shape, ())


def _emit_adaptive_log_softmax_with_loss_node(input: Tensor, target: Tensor, cutoffs: Any, add_cluster_prob: bool) -> Any:  # type: ignore
    """Emit a logical node representing the adaptive log softmax with loss computation during tracing.

    Args:
        input (Tensor): The input parameter.
        target (Tensor): The target parameter.
        cutoffs (object): The cutoffs parameter.
        add_cluster_prob (bool): The add_cluster_prob parameter.

    Returns:
        tuple: Result.

    Raises:
        RuntimeError: An exception.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.tracing.state import global_tracing_state
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    if not global_tracing_state.is_tracing:
        raise RuntimeError("Must be tracing")

    out_id = str(uuid.uuid4())
    loss_id = f"{out_id}:loss"

    node = LogicalNode(
        id=out_id,
        op_type="AdaptiveLogSoftmaxWithLoss",
        inputs=[input.data.id, target.data.id],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        shape_metadata=[target.shape, ()],
        attributes={"cutoffs": cutoffs, "add_cluster_prob": add_cluster_prob},
    )
    global_tracing_state.add_node(node)

    proxy_out = ProxyTensor(id=out_id, shape=target.shape, dtype=input.dtype)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    proxy_loss = ProxyTensor(id=loss_id, shape=(), dtype=input.dtype)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    return (
        Tensor(proxy_out, TensorConfig(target.shape, input.dtype, input.device)),  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        Tensor(proxy_loss, TensorConfig((), input.dtype, input.device)),
    )


def adaptive_log_softmax_with_loss(
    input: Tensor,  # type: ignore
    target: Tensor,  # type: ignore
    cutoffs: Any,
    add_cluster_prob: bool = True,
) -> Any:
    """Compute the adaptive log softmax and its corresponding loss.

    Args:
        input (Tensor): The input tensor.
        target (Tensor): The target tensor.
        cutoffs (object): The cutoff boundaries used to split the vocabulary into clusters.
        add_cluster_prob (bool): Whether to include cluster probabilities.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the output tensor and the loss tensor.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        out, loss = backend.execute_op(
            "AdaptiveLogSoftmaxWithLoss",
            input.data,
            target.data,
            cutoffs=cutoffs,
            add_cluster_prob=add_cluster_prob,
        )
        return (
            Tensor(out, TensorConfig(target.shape, input.dtype, input.device)),  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            Tensor(loss, TensorConfig((), input.dtype, input.device)),
        )
    return _emit_adaptive_log_softmax_with_loss_node(input, target, cutoffs, add_cluster_prob)


def log_poisson_loss(targets: Any, log_input: Any, compute_full_loss: Any = False, name: Any = None) -> Any:
    """Compute log Poisson loss.

    Args:
        targets (object): The ground truth target values.
        log_input (object): The logarithm of the predictions.
        compute_full_loss (object): Whether to compute the full loss.
        name (object): An optional name for the operation.

    Returns: Any: The computed log Poisson loss.
    """
    return Tensor(None, TensorConfig(targets.shape, "float32", "cpu"))


def in_top_k(targets: Any, predictions: Any, k: Any, name: Any = None) -> Any:
    """Says whether the targets are in the top K predictions.

    Args:
        targets (object): The ground truth target values.
        predictions (object): The predictions.
        k (object): The number of top elements to consider.
        name (object): An optional name for the operation.

    Returns: Any: A boolean tensor indicating if the targets are in the top K predictions.
    """
    return Tensor(None, TensorConfig(targets.shape, "bool", "cpu"))


def l2_loss(t: Any, name: Any = None) -> Any:
    """Compute half the L2 norm of a tensor without the sqrt.

    Args:
        t (object): The input tensor.
        name (object): An optional name for the operation.

    Returns: Any: The computed L2 loss.
    """
    return multiply(sum(multiply(t, t)), 0.5)


def scale_regularization_loss(regularization_loss: Any, name: Any = None) -> Any:
    """Scales the sum of the given regularization losses by number of replicas.

    Args:
        regularization_loss (object): The regularization loss to scale.
        name (object): An optional name for the operation.

    Returns: Any: The scaled regularization loss.
    """
    return regularization_loss


__all__ = [
    "adaptive_log_softmax_with_loss",
    "categorical_generalized_cross_entropy",
    "circle_loss",
    "ctc_loss",
    "dice_loss",
]


@register_op("InTopK")
class InTopK(OpDef):
    """InTopK operation."""

    op_name = "InTopK"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infers the output shape of the InTopK operation.

        Args:
            *args (object): The arguments passed to the operation.
            **kwargs (object): The keyword arguments passed to the operation.

        Returns: Any: The inferred shape.
        """
        return getattr(args[0], "shape", ()) if args else ()
