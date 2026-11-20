"""Generic backend-parametric eager reference operations for `nn_ops.losses`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendArrayProtocol,
    coerce_array,
    coerce_float,
    coerce_int,
    coerce_operand_list,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _np_ctc_loss_update_alpha(
    t: int,
    S: int,
    augmented: BackendArrayProtocol,
    alpha: BackendArrayProtocol,
    probs: BackendArrayProtocol,
    num_classes: int,
    blank: int,
) -> None:
    """Update CTC loss alpha matrix.

    Args:
        t (int): Time step.
        S (int): Sequence length.
        augmented (BackendArrayProtocol): The augmented labels.
        alpha (BackendArrayProtocol): The alpha matrix.
        probs (BackendArrayProtocol): The probabilities.
        num_classes (int): Number of classes.
        blank (int): Blank token index.
    """
    aug_t = augmented
    probs_t = probs
    alpha_t = alpha
    for s in range(S):
        prob_term = probs_t[t, aug_t[s] % num_classes]
        sum_term: BackendArrayProtocol | float = alpha_t[t - 1, s]
        if s > 0:
            sum_term = sum_term + alpha_t[t - 1, s - 1]
        if s >= 2 and aug_t[s] != blank and aug_t[s] != aug_t[s - 2]:
            sum_term = sum_term + alpha_t[t - 1, s - 2]
        alpha_t[t, s] = sum_term * prob_term


def _np_ctc_loss_single(probs: BackendArg, b_labels: BackendArg, T: int, L: int) -> float:
    """Help to compute the CTC loss for a single sequence.

    Args:
        probs (BackendArg): Probability distribution of logits over time.
        b_labels (BackendArg): Target labels for the current sequence.
        T (int): The logit length.
        L (int): The label length.

    Returns:
        float: Computed negative log-probability loss.
    """
    arr_mod = _resolve_eager_mod(None)
    blank = 0
    augmented = arr_mod.zeros(2 * L + 1, dtype=arr_mod.int32)
    b_labels_list = coerce_operand_list(b_labels)
    if len(b_labels_list) > 0:
        augmented[1::2] = arr_mod.asarray(b_labels, dtype=arr_mod.int32)
    S = len(augmented)

    probs_t = coerce_array(arr_mod, probs)
    num_classes = probs_t.shape[-1]
    alpha = arr_mod.zeros((T, S))
    if T > 0:
        alpha[0, 0] = probs_t[0, blank % num_classes]
        if S > 1:
            alpha[0, 1] = probs_t[0, augmented[1] % num_classes]

    for t in range(1, T):
        _np_ctc_loss_update_alpha(t, S, augmented, alpha, probs_t, num_classes, blank)

    p_total: BackendArrayProtocol | float = alpha[T - 1, S - 1] if S > 0 else 0.0
    if S > 1:
        p_total = p_total + alpha[T - 1, S - 2]

    return float(-arr_mod.log(arr_mod.maximum(p_total, 1e-30)))


@global_eager_registry.register("CtcLoss")
def _np_ctc_loss(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    labels: BackendArg,
    logits: BackendArg,
    label_length: BackendArg,
    logit_length: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np ctc loss operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        labels (BackendArg): Input `labels` argument (labels).
        logits (BackendArg): Input `logits` argument (logits).
        label_length (BackendArg): Input `label_length` argument (label length).
        logit_length (BackendArg): Input `logit_length` argument (logit length).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    logits_arr = arr_mod.asarray(logits)
    labels_arr = arr_mod.asarray(labels)
    label_lengths = arr_mod.atleast_1d(arr_mod.asarray(label_length))
    logit_lengths = arr_mod.atleast_1d(arr_mod.asarray(logit_length))

    time_major = kwargs.get("logits_time_major", True)
    if logits_arr.ndim == 1:
        logits_arr = arr_mod.expand_dims(logits_arr, axis=(1, 2))
    elif logits_arr.ndim == 2:
        logits_arr = arr_mod.expand_dims(logits_arr, axis=1)

    if not time_major:
        logits_arr = arr_mod.transpose(logits_arr, (1, 0, 2))

    max_time, batch_size, num_classes = logits_arr.shape

    if labels_arr.ndim == 1:
        labels_arr = arr_mod.expand_dims(labels_arr, axis=0)

    losses = []

    for b in range(batch_size):
        T = int(logit_lengths[b]) if b < len(logit_lengths) else max_time
        L = int(label_lengths[b]) if b < len(label_lengths) else labels_arr.shape[1]
        b_labels = labels_arr[b, :L] if b < len(labels_arr) else labels_arr[0, :L]

        logits_b = logits_arr[:T, b, :]
        max_logits = arr_mod.max(logits_b, axis=-1, keepdims=True)
        exp_logits = arr_mod.exp(logits_b - max_logits)
        probs = exp_logits / arr_mod.sum(exp_logits, axis=-1, keepdims=True)

        loss = _np_ctc_loss_single(probs, b_labels, T, L)
        losses.append(loss)

    return arr_mod.array(losses, dtype=arr_mod.float32)


@global_eager_registry.register("CircleLoss")
def _np_circle_loss(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np circle loss operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if len(args) < 2:
        return adapt_backend_module(backend_module).require_symbol("zeros")(1)
    y_true = arr_mod.asarray(args[0])
    y_pred = arr_mod.asarray(args[1])
    margin = coerce_float(kwargs.get("margin", 0.25))
    gamma = coerce_float(kwargs.get("gamma", 256.0))

    O_p = 1 + margin
    O_n = -margin
    Delta_p = 1 - margin
    Delta_n = margin

    alpha_p = adapt_backend_module(arr_mod).require_symbol("maximum")(0, O_p - y_pred)
    alpha_n = arr_mod.maximum(0, y_pred - O_n)

    p_loss = alpha_p * (y_pred - Delta_p)
    n_loss = alpha_n * (y_pred - Delta_n)

    # Positive and negative masks based on y_true (assuming binary or one-hot)
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    logit_p = -gamma * p_loss
    logit_n = gamma * n_loss

    # Use logaddexp for numerical stability
    # L = log(1 + sum(exp(logit_n)) * sum(exp(logit_p)))

    sum_exp_n = arr_mod.sum(arr_mod.exp(logit_n * neg_mask) * neg_mask, axis=-1, keepdims=True)
    sum_exp_p = arr_mod.sum(arr_mod.exp(logit_p * pos_mask) * pos_mask, axis=-1, keepdims=True)

    loss = arr_mod.log1p(sum_exp_n * sum_exp_p)
    return arr_mod.mean(loss)


@global_eager_registry.register("CategoricalGeneralizedCrossEntropy")
def _np_categorical_generalized_cross_entropy(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np categorical generalized cross entropy operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if len(args) < 2:
        return adapt_backend_module(backend_module).require_symbol("zeros")(1)
    y_true = arr_mod.asarray(args[0])
    y_pred = arr_mod.asarray(args[1])
    q = coerce_float(kwargs.get("q", 0.9))
    y_pred = arr_mod.clip(y_pred, 1e-7, 1.0)

    # GCE = (1 - sum(y_true * y_pred^q)) / q
    res = (1.0 - arr_mod.sum(y_true * adapt_backend_module(arr_mod).require_symbol("power")(y_pred, q), axis=-1)) / q
    return arr_mod.mean(res)


@global_eager_registry.register("InTopK")
def _np_in_top_k(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    targets: BackendArg,
    predictions: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Evaluate InTopK operation in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): NumPy backend module.
        targets (BackendArg): Target class indices of shape (batch_size,) or (batch_size, ...).
        predictions (BackendArg): Prediction scores of shape (batch_size, num_classes).
        **kwargs (BackendArg): Keyword arguments including 'k'.

    Returns:
        BackendArg: Boolean array indicating if target is within top K predictions.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    _ = backend_module
    k = int(coerce_int(kwargs.get("k", 1)))
    t_arr = arr_mod.asarray(targets, dtype=arr_mod.int64).reshape(-1)
    p_arr = arr_mod.atleast_2d(arr_mod.asarray(predictions))
    sorted_indices = arr_mod.argsort(-p_arr, axis=-1)[:, : max(1, k)]
    result = [bool(t_arr[i] in sorted_indices[i % len(sorted_indices)]) for i in range(len(t_arr))]
    return arr_mod.array(result, dtype=bool).reshape(arr_mod.asarray(targets).shape)
