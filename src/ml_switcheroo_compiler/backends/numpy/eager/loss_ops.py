"""Numpy Loss Ops."""

# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _np_ctc_loss_update_alpha(t: int, S: int, augmented: np.ndarray, alpha: np.ndarray, probs: np.ndarray, num_classes: int, blank: int) -> None:
    """Update CTC loss alpha matrix.

    Args:
        alpha (object): The alpha matrix.
        probs (object): The probabilities.
        augmented (object): The augmented labels.
        t (int): Time step.
        S (int): Sequence length.
        blank (int): Blank token index.
        num_classes (int): Number of classes.
    """
    for s in range(S):
        prob_term = probs[t, augmented[s] % num_classes]
        sum_term = alpha[t - 1, s]
        if s > 0:
            sum_term += alpha[t - 1, s - 1]
        if s >= 2 and augmented[s] != blank and augmented[s] != augmented[s - 2]:
            sum_term += alpha[t - 1, s - 2]
        alpha[t, s] = sum_term * prob_term


def _np_ctc_loss_single(probs: np.ndarray, b_labels: np.ndarray, T: int, L: int) -> float:
    """Help to compute the CTC loss for a single sequence.

    Args:
        probs (np.ndarray): Probability distribution of logits over time.
        b_labels (np.ndarray): Target labels for the current sequence.
        T (int): The logit length.
        L (int): The label length.

    Returns:
        float: Computed negative log-probability loss.
    """
    blank = 0
    augmented = np.zeros(2 * L + 1, dtype=np.int32)
    if len(b_labels) > 0:
        augmented[1::2] = b_labels
    S = len(augmented)

    num_classes = probs.shape[-1]
    alpha = np.zeros((T, S))
    if T > 0:
        alpha[0, 0] = probs[0, blank % num_classes]
        if S > 1:
            alpha[0, 1] = probs[0, augmented[1] % num_classes]

    for t in range(1, T):
        _np_ctc_loss_update_alpha(t, S, augmented, alpha, probs, num_classes, blank)

    p_total = alpha[T - 1, S - 1] if S > 0 else 0.0
    if S > 1:
        p_total += alpha[T - 1, S - 2]

    return float(-np.log(np.maximum(p_total, 1e-30)))


@numpy_eager_registry.register("CtcLoss")
def _np_ctc_loss(backend_module: object, labels: object, logits: object, label_length: object, logit_length: object, **kwargs: object) -> object:
    """Evaluate _np_ctc_loss operation.

    Args:
        backend_module (object): The backend_module parameter.
        labels (object): The labels parameter.
        logits (object): The logits parameter.
        label_length (object): The label_length parameter.
        logit_length (object): The logit_length parameter.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    logits_arr = np.asarray(logits)
    labels_arr = np.asarray(labels)
    label_lengths = np.atleast_1d(np.asarray(label_length))
    logit_lengths = np.atleast_1d(np.asarray(logit_length))

    time_major = kwargs.get("logits_time_major", True)
    if logits_arr.ndim == 1:
        logits_arr = np.expand_dims(logits_arr, axis=(1, 2))
    elif logits_arr.ndim == 2:
        logits_arr = np.expand_dims(logits_arr, axis=1)

    if not time_major:
        logits_arr = np.transpose(logits_arr, (1, 0, 2))

    max_time, batch_size, num_classes = logits_arr.shape

    if labels_arr.ndim == 1:
        labels_arr = np.expand_dims(labels_arr, axis=0)

    losses = []

    for b in range(batch_size):
        T = int(logit_lengths[b]) if b < len(logit_lengths) else max_time
        L = int(label_lengths[b]) if b < len(label_lengths) else labels_arr.shape[1]
        b_labels = labels_arr[b, :L] if b < len(labels_arr) else labels_arr[0, :L]

        logits_b = logits_arr[:T, b, :]
        max_logits = np.max(logits_b, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_b - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        loss = _np_ctc_loss_single(probs, b_labels, T, L)
        losses.append(loss)

    return np.array(losses, dtype=np.float32)


@numpy_eager_registry.register("CircleLoss")
def _np_circle_loss(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_circle_loss operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    if len(args) < 2:
        return backend_module.zeros(1)
    y_true = np.asarray(args[0])
    y_pred = np.asarray(args[1])
    margin = kwargs.get("margin", 0.25)
    gamma = kwargs.get("gamma", 256.0)

    O_p = 1 + margin
    O_n = -margin
    Delta_p = 1 - margin
    Delta_n = margin

    alpha_p = np.maximum(0, O_p - y_pred)
    alpha_n = np.maximum(0, y_pred - O_n)

    p_loss = alpha_p * (y_pred - Delta_p)
    n_loss = alpha_n * (y_pred - Delta_n)

    # Positive and negative masks based on y_true (assuming binary or one-hot)
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    logit_p = -gamma * p_loss
    logit_n = gamma * n_loss

    # Use logaddexp for numerical stability
    # L = log(1 + sum(exp(logit_n)) * sum(exp(logit_p)))

    sum_exp_n = np.sum(np.exp(logit_n * neg_mask) * neg_mask, axis=-1, keepdims=True)
    sum_exp_p = np.sum(np.exp(logit_p * pos_mask) * pos_mask, axis=-1, keepdims=True)

    loss = np.log1p(sum_exp_n * sum_exp_p)
    return np.mean(loss)


@numpy_eager_registry.register("CategoricalGeneralizedCrossEntropy")
def _np_categorical_generalized_cross_entropy(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_categorical_generalized_cross_entropy operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    if len(args) < 2:
        return backend_module.zeros(1)
    y_true = np.asarray(args[0])
    y_pred = np.asarray(args[1])
    q = kwargs.get("q", 0.9)
    y_pred = np.clip(y_pred, 1e-7, 1.0)

    # GCE = (1 - sum(y_true * y_pred^q)) / q
    res = (1.0 - np.sum(y_true * np.power(y_pred, q), axis=-1)) / q
    return np.mean(res)
