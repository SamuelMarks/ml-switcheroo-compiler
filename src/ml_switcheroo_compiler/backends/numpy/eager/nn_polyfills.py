"""Module nn_polyfills.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy eager fallback implementations for stubbed NN operations."""
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("LogPoissonLoss")
def _np_log_poisson_loss(backend_module: Any, targets: Any, log_input: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_log_poisson_loss.

    Args:
        backend_module (object): The backend_module parameter.
        targets (object): The targets parameter.
        log_input (object): The log_input parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    t = np.asarray(targets)
    log_inp = np.asarray(log_input)
    compute_full = kwargs.get("compute_full_loss", False)
    res = np.exp(log_inp) - t * log_inp
    if compute_full:
        import scipy.special

        res += scipy.special.gammaln(t + 1)
    return res


@numpy_eager_registry.register("AllCandidateSampler")
def _np_all_candidate_sampler(backend_module: Any, true_classes: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Eager fallback for _np_all_candidate_sampler.

    Args:
        backend_module (object): The backend_module parameter.
        true_classes (object): The true_classes parameter.
        **kwargs (object): Keyword args.

    Returns:
        tuple: Result.
    """
    t = np.asarray(true_classes)
    num_sampled = kwargs.get("num_sampled", 1)
    num_classes = kwargs.get("num_classes", 10)
    sampled = np.random.choice(num_classes, num_sampled, replace=True)
    true_expected_count = np.ones_like(t, dtype=np.float32) * (num_sampled / num_classes)
    sampled_expected_count = np.ones((num_sampled,), dtype=np.float32) * (num_sampled / num_classes)
    return sampled.astype(np.int32), true_expected_count, sampled_expected_count


def _np_ctc_beam_step(beam: dict[tuple[int, ...], tuple[float, float]], log_p: Any, num_classes: int, blank: int, beam_width: int) -> dict[tuple[int, ...], tuple[float, float]]:
    """Eager fallback for _np_ctc_beam_step.

    Args:
        beam (dict): The beam parameter.
        log_p (object): The log_p parameter.
        num_classes (int): The num_classes parameter.
        blank (int): The blank parameter.
        beam_width (int): The beam_width parameter.

    Returns:
        dict: Result.
    """
    next_beam: dict[tuple[int, ...], tuple[float, float]] = {}
    for path, (p_b, p_nb) in beam.items():
        p_tot = np.logaddexp(p_b, p_nb)

        # Case 1: extension with blank
        n_p_b = p_tot + log_p[blank]
        if path in next_beam:
            next_beam[path] = (np.logaddexp(next_beam[path][0], n_p_b), next_beam[path][1])
        else:
            next_beam[path] = (n_p_b, -float("inf"))

        # Case 2: extension with non-blank
        for c in range(num_classes - 1):
            n_path = path + (c,)
            c_log_p = log_p[c]

            if len(path) > 0 and path[-1] == c:
                # same char, needs a blank in between to be separated
                # path extended by same char: comes from blank
                n_p_nb = p_b + c_log_p

                # path not extended (same char absorbed)
                n_p_nb_keep = p_nb + c_log_p

                next_beam[path] = (next_beam[path][0], np.logaddexp(next_beam[path][1], n_p_nb_keep))
            else:
                n_p_nb = p_tot + c_log_p

            if n_path in next_beam:
                next_beam[n_path] = (next_beam[n_path][0], np.logaddexp(next_beam[n_path][1], n_p_nb))
            else:
                next_beam[n_path] = (-float("inf"), n_p_nb)

    # Prune beam
    sorted_beam = sorted(next_beam.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]), reverse=True)
    return dict(sorted_beam[:beam_width])


@numpy_eager_registry.register("CtcBeamSearchDecoder")
def _np_ctc_beam_search_decoder(backend_module: Any, inputs: Any, sequence_length: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Eager fallback for _np_ctc_beam_search_decoder.

    Args:
        backend_module (object): The backend_module parameter.
        inputs (object): The inputs parameter.
        sequence_length (object): The sequence_length parameter.
        **kwargs (object): Keyword args.

    Returns:
        tuple: Result.
    """
    arr = np.asarray(inputs)
    seq_len = np.asarray(sequence_length)
    beam_width = kwargs.get("beam_width", 100)
    top_paths = kwargs.get("top_paths", 1)

    # arr is [max_time, batch_size, num_classes]
    # We will compute log probabilities.
    # To keep it simple and correct without numerical underflow, we should work in log space.
    # But for a numpy fallback, a simple beam search over the batch.

    batch_size = arr.shape[1]
    num_classes = arr.shape[2]
    blank = num_classes - 1

    decoded = []
    log_probs = []

    for b in range(batch_size):
        T = seq_len[b]
        # beam: dictionary of {tuple(path): (log_prob_blank, log_prob_non_blank)}
        # Initialize with empty path
        beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -float("inf"))}

        for t in range(T):
            probs = arr[t, b]
            # Convert to log probs if they aren't already. Assuming they are logits.
            # But the TF spec says inputs are logits.
            max_p: Any = np.max(probs)
            log_p = probs - max_p - np.log(np.sum(np.exp(probs - max_p)))

            beam = _np_ctc_beam_step(beam, log_p, num_classes, blank, beam_width)

        # Select top paths
        best_paths = sorted(beam.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]), reverse=True)

        # for a simple fallback, we just take top_paths=1 for now.
        best_path = best_paths[0][0] if best_paths else ()
        best_prob = np.logaddexp(best_paths[0][1][0], best_paths[0][1][1]) if best_paths else 0.0

        decoded.append(best_path)
        log_probs.append(best_prob)

    indices = []
    values = []
    for b, seq in enumerate(decoded):
        for t, val in enumerate(seq):
            indices.append([b, t])
            values.append(val)

    if len(indices) == 0:
        indices = np.zeros((0, 2), dtype=np.int64)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    sparse = (np.array(indices, dtype=np.int64), np.array(values, dtype=np.int64), np.array([batch_size, max((len(s) for s in decoded), default=0)], dtype=np.int64))

    # Usually it returns a list of sparse tensors if top_paths > 1, but we return a tuple containing the sparse tensor
    # depending on TF spec. We just return a tuple (sparse_tensor, log_probs)
    return sparse, np.array(log_probs, dtype=np.float32)


@numpy_eager_registry.register("CtcUniqueLabels")
def _np_ctc_unique_labels(backend_module: Any, labels: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Eager fallback for _np_ctc_unique_labels.

    Args:
        backend_module (object): The backend_module parameter.
        labels (object): The labels parameter.
        **kwargs (object): Keyword args.

    Returns:
        tuple: Result.
    """
    arr = np.asarray(labels)
    unique, indices = np.unique(arr, return_inverse=True)
    return unique.astype(np.int32), indices.astype(np.int32)


@numpy_eager_registry.register("NormalizeMoments")
def _np_normalize_moments(backend_module: Any, counts: Any, mean_ss: Any, variance_ss: Any, shift: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Eager fallback for _np_normalize_moments.

    Args:
        backend_module (object): The backend_module parameter.
        counts (object): The counts parameter.
        mean_ss (object): The mean_ss parameter.
        variance_ss (object): The variance_ss parameter.
        shift (object): The shift parameter.
        **kwargs (object): Keyword args.

    Returns:
        tuple: Result.
    """
    c = np.asarray(counts)
    m = np.asarray(mean_ss)
    v = np.asarray(variance_ss)
    s = np.asarray(shift)
    mean = m / np.maximum(c, 1e-10) + s
    variance = np.maximum(v / np.maximum(c, 1e-10) - np.square(m / np.maximum(c, 1e-10)), 0.0)
    return mean, variance


@numpy_eager_registry.register("SufficientStatistics")
def _np_sufficient_statistics(backend_module: Any, x: Any, axes: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Eager fallback for _np_sufficient_statistics.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axes (object): The axes parameter.
        **kwargs (object): Keyword args.

    Returns:
        tuple: Result.
    """
    arr = np.asarray(x)
    axes = tuple(np.asarray(axes).tolist()) if axes is not None else None
    divisor = np.prod([arr.shape[i] for i in range(arr.ndim) if i not in (axes or [])]) if axes else 1
    counts = np.array(arr.size / divisor, dtype=np.float32)
    counts = np.broadcast_to(counts, np.mean(arr, axis=axes, keepdims=kwargs.get("keepdims", False)).shape)
    shift = np.mean(arr, axis=axes, keepdims=True)
    m_ss = np.sum(arr - shift, axis=axes, keepdims=kwargs.get("keepdims", False))
    v_ss = np.sum(np.square(arr - shift), axis=axes, keepdims=kwargs.get("keepdims", False))
    return counts, m_ss, v_ss, np.squeeze(shift) if not kwargs.get("keepdims", False) else shift


@numpy_eager_registry.register("WeightedMoments")
def _np_weighted_moments(backend_module: Any, x: Any, axes: Any, frequency_weights: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Eager fallback for _np_weighted_moments.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axes (object): The axes parameter.
        frequency_weights (object): The frequency_weights parameter.
        **kwargs (object): Keyword args.

    Returns:
        tuple: Result.
    """
    arr = np.asarray(x)
    fw = np.asarray(frequency_weights)
    axes = tuple(np.asarray(axes).tolist()) if axes is not None else None
    keepdims = kwargs.get("keepdims", False)
    sum_w = np.sum(fw, axis=axes, keepdims=keepdims)
    mean = np.sum(arr * fw, axis=axes, keepdims=keepdims) / np.maximum(sum_w, 1e-10)
    var = np.sum(fw * np.square(arr - mean if keepdims else np.expand_dims(mean, axes)), axis=axes, keepdims=keepdims) / np.maximum(sum_w, 1e-10)
    return mean, var


@numpy_eager_registry.register("MaxPoolWithArgmax")
def _np_max_pool_with_argmax(backend_module: Any, input: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Eager fallback for _np_max_pool_with_argmax.

    Args:
        backend_module (object): The backend_module parameter.
        input (object): The input parameter.
        **kwargs (object): Keyword args.

    Returns:
        tuple: Result.
    """
    arr = np.asarray(input)
    import scipy.ndimage

    size = kwargs.get("pool_size", (2, 2))
    if isinstance(size, int):
        size = (1, size, size, 1) if arr.ndim == 4 else (size, size)
    maxes = scipy.ndimage.maximum_filter(arr, size=size)
    argmaxes = np.argmax(arr.reshape(arr.shape[0], -1), axis=1)
    return maxes, argmaxes.astype(np.int32)


@numpy_eager_registry.register("CollapseRepeated")
def _np_collapse_repeated(backend_module: Any, labels: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Eager fallback for _np_collapse_repeated.

    Args:
        backend_module (object): The backend_module parameter.
        labels (object): The labels parameter.
        **kwargs (object): Keyword args.

    Returns:
        tuple: Result.
    """
    arr = np.asarray(labels)
    if arr.size == 0:
        return arr, np.zeros_like(arr, dtype=np.int32)
    mask = np.ones(len(arr), dtype=bool)
    mask[1:] = arr[1:] != arr[:-1]
    collapsed = arr[mask]
    return collapsed, np.arange(len(collapsed), dtype=np.int32)


@numpy_eager_registry.register("QuantizedConv")
def _np_quantized_conv(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """NumPy eager evaluation fallback for QuantizedConv.

    Args:
        backend_module (object): The active backend module.
        *args: Positional arguments containing input, weight, scales, and optional biases.
        **kwargs: Convolution hyperparameters (stride, padding, dilation, groups).

    Returns: Any: The result of the quantized convolution.
    """
    import numpy as np

    # 1. Unpack arguments
    input_val = np.asarray(args[0])
    weight_val = np.asarray(args[1])
    scales_val = np.asarray(args[2])
    if len(args) > 3 and args[3] is not None:
        biases_val = np.asarray(args[3])
    else:
        biases_val = np.zeros(1, dtype=np.float32)

    # 2. Dequantize weights
    w_float = (weight_val - biases_val) * scales_val

    # 3. Retrieve hyperparameters
    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)

    # 4. Invoke the standard ConvGeneralDilated eager implementation
    from ml_switcheroo_compiler.ops.configs import ConvConfig

    spatial_dims = w_float.ndim - 2

    # Map strides, padding, and dilation to XLA-style/ConvConfig format
    strides_tuple = (stride,) * spatial_dims if isinstance(stride, int) else tuple(stride)
    dilation_tuple = (dilation,) * spatial_dims if isinstance(dilation, int) else tuple(dilation)

    pad_arg: Any = padding
    if isinstance(padding, int):
        if padding == 0:
            pad_arg = "VALID"
        else:
            pad_arg = [(padding, padding)] * spatial_dims

    # XLA-style dimension specification for layout: NHWC and HWIO layouts
    lhs_spec = (0, spatial_dims + 1) + tuple(range(1, spatial_dims + 1))
    rhs_spec = (spatial_dims + 1, spatial_dims) + tuple(range(0, spatial_dims))
    out_spec = lhs_spec

    config_obj = ConvConfig(
        window_strides=strides_tuple,
        padding=pad_arg,
        lhs_dilation=(1,) * spatial_dims,
        rhs_dilation=dilation_tuple,
        dimension_numbers=(lhs_spec, rhs_spec, out_spec),
        feature_group_count=groups,
    )

    from ml_switcheroo_compiler.backends.numpy.eager.conv import _conv_general_dilated

    res = _conv_general_dilated(input_val, w_float, config_obj)
    return res
