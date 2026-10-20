"""Numpy eager fallback implementations for stubbed NN operations."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable


def _stirling_gammaln(x: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Vectorized Ramanujan/Stirling approximation of log(gamma(x)) for x > 0 in pure NumPy.

    Args:
        x (np.ndarray): Input values.

    Returns:
        np.ndarray: Approximated ln(Gamma(x)).
    """
    x_safe = np.maximum(x, 1e-15)
    c0 = 0.5 * np.log(2.0 * np.pi)
    res = (x_safe - 0.5) * np.log(x_safe) - x_safe + c0 + (1.0 / (12.0 * x_safe)) - (1.0 / (360.0 * np.power(x_safe, 3)))
    small_mask = x_safe < 1.5
    if np.any(small_mask):
        xs = x_safe[small_mask] + 1.0
        g_next = (xs - 0.5) * np.log(xs) - xs + c0 + (1.0 / (12.0 * xs)) - (1.0 / (360.0 * np.power(xs, 3)))
        res[small_mask] = g_next - np.log(x_safe[small_mask])
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], res)


@numpy_eager_registry.register("LogPoissonLoss")
def _np_log_poisson_loss(backend_module: BackendModuleProtocol, targets: EagerValue, predictions: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Numerically stable vectorized LogPoissonLoss computation.

    Args:
        backend_module (EagerValue): Active backend module.
        targets (EagerValue): Ground truth targets tensor.
        predictions (EagerValue): Predictions (log-scale if log_input=True).
        **kwargs (EagerValue): Hyperparameters including compute_full_loss and log_input.

    Returns:
        np.ndarray: Evaluated loss values.
    """
    t = np.asarray(targets)
    inp = np.asarray(predictions)
    is_log = kwargs.get("log_input", True)
    compute_full = kwargs.get("compute_full_loss", False)

    if is_log:
        res = np.exp(inp) - t * inp
    else:
        res = inp - t * np.log(np.maximum(inp, 1e-12))

    if compute_full:
        try:
            import scipy.special

            res = res + scipy.special.gammaln(t + 1.0)
        except ImportError:
            res = res + _stirling_gammaln(t + 1.0)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], res)


@numpy_eager_registry.register("AllCandidateSampler")
def _np_all_candidate_sampler(backend_module: BackendModuleProtocol, true_classes: EagerValue, **kwargs: EagerValue) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Numerically stable vectorized eager fallback for AllCandidateSampler.

    Args:
        backend_module (EagerValue): The backend module.
        true_classes (EagerValue): Ground truth class indices tensor.
        **kwargs (EagerValue): Sampling options (num_sampled, num_classes, unique, seed).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (sampled_candidates, true_expected_count, sampled_expected_count).
    """
    t = np.asarray(true_classes, dtype=np.int64)
    num_sampled = int(typing.cast(int, kwargs.get("num_sampled", 1)))
    num_classes = int(typing.cast(int, kwargs.get("num_classes", 10)))
    unique = bool(kwargs.get("unique", True))
    seed = kwargs.get("seed")

    rng = getattr(np.random, "default_rng")(seed)
    replace = not unique
    sampled = rng.choice(num_classes, size=num_sampled, replace=replace)

    prob = num_sampled / max(1, num_classes)
    true_expected_count = np.full(t.shape, prob, dtype=np.float32)
    sampled_expected_count = np.full((num_sampled,), prob, dtype=np.float32)
    return sampled.astype(np.int32), true_expected_count, sampled_expected_count


def _np_ctc_beam_step(
    beam: dict[tuple[int, ...], tuple[float, float]],
    log_p: EagerValue,
    num_classes: int,
    blank: int,
    beam_width: int,
) -> dict[tuple[int, ...], tuple[float, float]]:
    """Numerically stable vectorized CTC beam search step in log space.

    Args:
        beam (dict[tuple[int, ...], tuple[float, float]]): Current beam hypotheses mapping path -> (log_p_blank, log_p_non_blank).
        log_p (EagerValue): Log probability distribution over classes for current timestep.
        num_classes (int): Total number of target output classes including blank.
        blank (int): Index of the CTC blank token.
        beam_width (int): Beam capacity width threshold.

    Returns:
        dict[tuple[int, ...], tuple[float, float]]: Pruned beam for next timestep.
    """
    log_p_arr = np.asarray(log_p, dtype=np.float32)
    next_beam: dict[tuple[int, ...], tuple[float, float]] = {}

    for path, (p_b, p_nb) in beam.items():
        p_tot = float(np.logaddexp(p_b, p_nb))

        # Extension with blank
        n_p_b = p_tot + float(log_p_arr[blank])
        if path in next_beam:
            next_beam[path] = (float(np.logaddexp(next_beam[path][0], n_p_b)), next_beam[path][1])
        else:
            next_beam[path] = (n_p_b, -float("inf"))

        # Extension with non-blank characters
        for c in range(num_classes - 1):
            n_path = path + (c,)
            c_log_p = float(log_p_arr[c])

            if len(path) > 0 and path[-1] == c:
                n_p_nb = p_b + c_log_p
                n_p_nb_keep = p_nb + c_log_p
                prev_b, prev_nb = next_beam[path]
                next_beam[path] = (prev_b, float(np.logaddexp(prev_nb, n_p_nb_keep)))
            else:
                n_p_nb = p_tot + c_log_p

            if n_path in next_beam:
                prev_b, prev_nb = next_beam[n_path]
                next_beam[n_path] = (prev_b, float(np.logaddexp(prev_nb, n_p_nb)))
            else:
                next_beam[n_path] = (-float("inf"), n_p_nb)

    sorted_beam = sorted(next_beam.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]), reverse=True)
    return dict(sorted_beam[:beam_width])


@numpy_eager_registry.register("CtcBeamSearchDecoder")
def _np_ctc_beam_search_decoder(backend_module: BackendModuleProtocol, inputs: EagerValue, sequence_length: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_ctc_beam_search_decoder.

    Args:
        backend_module (EagerValue): The backend_module parameter.
        inputs (EagerValue): The inputs parameter.
        sequence_length (EagerValue): The sequence_length parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
        EagerValue: Result.
    """
    arr = np.asarray(inputs)
    seq_len = np.asarray(sequence_length)
    beam_width = kwargs.get("beam_width", 100)
    top_paths = kwargs.get("top_paths", 1)

    batch_size = arr.shape[1] if arr.ndim > 1 else 0
    num_classes = arr.shape[2] if arr.ndim > 2 else 0
    blank = num_classes - 1

    best_paths_by_batch: list[list[tuple[tuple[int, ...], tuple[float, float]]]] = []

    for b in range(batch_size):
        T = int(seq_len[b])
        beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -float("inf"))}

        for t in range(T):
            probs = arr[t, b]
            max_p = np.max(probs)
            log_p = probs - max_p - np.log(np.sum(np.exp(probs - max_p)))
            beam = _np_ctc_beam_step(beam, log_p, num_classes, blank, typing.cast("int", beam_width))

        best_paths = sorted(beam.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]), reverse=True)
        best_paths_by_batch.append(best_paths)

    sparse_list: list[tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]] = []
    log_probs_matrix = getattr(np, "zeros")((batch_size, top_paths), dtype=np.float32)

    for p in range(typing.cast("int", top_paths)):
        indices_p: list[list[int]] = []
        values_p: list[int] = []
        max_len = 0
        for b in range(batch_size):
            paths_b = best_paths_by_batch[b]
            seq = paths_b[p][0] if p < len(paths_b) else ()
            prob = np.logaddexp(paths_b[p][1][0], paths_b[p][1][1]) if p < len(paths_b) else -float("inf")
            log_probs_matrix[b, p] = float(prob)
            if len(seq) > max_len:
                max_len = len(seq)
            for t, val in enumerate(seq):
                indices_p.append([b, t])
                values_p.append(val)

        ind_arr = np.array(indices_p, dtype=np.int64) if indices_p else np.zeros((0, 2), dtype=np.int64)
        val_arr = np.array(values_p, dtype=np.int64)
        shape_arr = np.array([batch_size, max_len], dtype=np.int64)
        sparse_list.append((ind_arr, val_arr, shape_arr))

    if top_paths == 1:
        if not sparse_list:
            return (np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((2,), dtype=np.int64)), log_probs_matrix[:, 0]
        return sparse_list[0], log_probs_matrix[:, 0]
    return sparse_list, log_probs_matrix


@numpy_eager_registry.register("CtcUniqueLabels")
def _np_ctc_unique_labels(backend_module: BackendModuleProtocol, labels: EagerValue, **kwargs: EagerValue) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Eager fallback for _np_ctc_unique_labels.

    Args:
        backend_module (EagerValue): The backend_module parameter.
        labels (EagerValue): The labels parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
        tuple[np.ndarray, np.ndarray]: Result.
    """
    arr = np.asarray(labels)
    unique, indices = np.unique(arr, return_inverse=True)
    return unique.astype(np.int32), indices.astype(np.int32)


@numpy_eager_registry.register("NormalizeMoments")
def _np_normalize_moments(backend_module: BackendModuleProtocol, counts: EagerValue, mean_ss: EagerValue, variance_ss: EagerValue, shift: EagerValue, **kwargs: EagerValue) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Eager fallback for _np_normalize_moments.

    Args:
        backend_module (EagerValue): The backend_module parameter.
        counts (EagerValue): The counts parameter.
        mean_ss (EagerValue): The mean_ss parameter.
        variance_ss (EagerValue): The variance_ss parameter.
        shift (EagerValue): The shift parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
        tuple[np.ndarray, np.ndarray]: Result.
    """
    c = np.asarray(counts)
    m = np.asarray(mean_ss)
    v = np.asarray(variance_ss)
    s = np.asarray(shift)
    mean = m / np.maximum(c, 1e-10) + s
    variance = np.maximum(v / np.maximum(c, 1e-10) - np.square(m / np.maximum(c, 1e-10)), 0.0)
    return mean, variance


@numpy_eager_registry.register("SufficientStatistics")
def _np_sufficient_statistics(
    backend_module: BackendModuleProtocol, x: EagerValue, axes: EagerValue, **kwargs: EagerValue
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Eager fallback for _np_sufficient_statistics.

    Args:
        backend_module (EagerValue): The backend_module parameter.
        x (EagerValue): The x parameter.
        axes (EagerValue): The axes parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Result.
    """
    arr = np.asarray(x)
    axes_tup: tuple[int, ...] | None = tuple(np.asarray(axes).tolist()) if axes is not None else None
    keepdims = bool(kwargs.get("keepdims", False))
    divisor = np.prod([arr.shape[i] for i in range(arr.ndim) if i not in (axes_tup or ())]) if axes_tup else 1
    counts = np.array(arr.size / divisor, dtype=np.float32)
    counts = np.broadcast_to(counts, np.mean(arr, axis=axes_tup, keepdims=keepdims).shape)
    shift = np.mean(arr, axis=axes_tup, keepdims=True)
    m_ss = np.sum(arr - shift, axis=axes_tup, keepdims=keepdims)
    v_ss = np.sum(np.square(arr - shift), axis=axes_tup, keepdims=keepdims)
    return counts, m_ss, v_ss, np.squeeze(shift) if not keepdims else shift


@numpy_eager_registry.register("WeightedMoments")
def _np_weighted_moments(backend_module: BackendModuleProtocol, x: EagerValue, axes: EagerValue, frequency_weights: EagerValue, **kwargs: EagerValue) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Eager fallback for _np_weighted_moments.

    Args:
        backend_module (EagerValue): The backend_module parameter.
        x (EagerValue): The x parameter.
        axes (EagerValue): The axes parameter.
        frequency_weights (EagerValue): The frequency_weights parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
        tuple[np.ndarray, np.ndarray]: Result.
    """
    arr = np.asarray(x)
    fw = np.asarray(frequency_weights)
    axes_tup: tuple[int, ...] | None = tuple(np.asarray(axes).tolist()) if axes is not None else None
    keepdims = bool(kwargs.get("keepdims", False))
    sum_w = np.sum(fw, axis=axes_tup, keepdims=keepdims)
    mean = np.sum(arr * fw, axis=axes_tup, keepdims=keepdims) / np.maximum(sum_w, 1e-10)
    var = np.sum(fw * np.square(arr - mean if keepdims else np.expand_dims(mean, axes_tup or 0)), axis=axes_tup, keepdims=keepdims) / np.maximum(sum_w, 1e-10)
    return mean, var


@numpy_eager_registry.register("MaxPoolWithArgmax")
def _np_max_pool_with_argmax(backend_module: BackendModuleProtocol, input: EagerValue, **kwargs: EagerValue) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Eager fallback for _np_max_pool_with_argmax.

    Args:
        backend_module (EagerValue): The backend_module parameter.
        input (EagerValue): The input parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
        tuple[np.ndarray, np.ndarray]: Result.
    """
    arr = np.asarray(input)
    try:
        import scipy.ndimage

        size = kwargs.get("pool_size", (2, 2))
        if isinstance(size, int):
            size = (1, size, size, 1) if arr.ndim == 4 else (size, size)
        maxes = scipy.ndimage.maximum_filter(arr, size=size)
        argmaxes = np.argmax(arr.reshape(arr.shape[0], -1), axis=1)
        return maxes, argmaxes.astype(np.int32)
    except ImportError:
        argmaxes = np.argmax(arr.reshape(arr.shape[0], -1), axis=1)
        return arr, argmaxes.astype(np.int32)


@numpy_eager_registry.register("CollapseRepeated")
def _np_collapse_repeated(backend_module: BackendModuleProtocol, labels: EagerValue, **kwargs: EagerValue) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Eager fallback for _np_collapse_repeated.

    Args:
        backend_module (EagerValue): The backend_module parameter.
        labels (EagerValue): The labels parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
        tuple[np.ndarray, np.ndarray]: Result.
    """
    arr = np.asarray(labels)
    if arr.size == 0:
        return arr, np.zeros_like(arr, dtype=np.int32)
    mask = np.ones(len(arr), dtype=bool)
    mask[1:] = arr[1:] != arr[:-1]
    collapsed = arr[mask]
    return collapsed, np.arange(len(collapsed), dtype=np.int32)


@numpy_eager_registry.register("QuantizedConv")
def _np_quantized_conv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """NumPy eager evaluation fallback for QuantizedConv.

    Args:
        backend_module (EagerValue): The active backend module.
        *args (EagerValue): Positional arguments containing input, weight, scales, and optional biases.
        **kwargs (EagerValue): Convolution hyperparameters (stride, padding, dilation, groups).

    Returns:
        np.ndarray: The result of the quantized convolution.
    """
    input_val = np.asarray(args[0])
    weight_val = np.asarray(args[1])
    scales_val = np.asarray(args[2])
    if len(args) > 3 and args[3] is not None:
        biases_val = np.asarray(args[3])
    else:
        biases_val = np.zeros(1, dtype=np.float32)

    w_float = (weight_val - biases_val) * scales_val

    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)

    from ml_switcheroo_compiler.ops.configs import ConvConfig

    spatial_dims = w_float.ndim - 2
    strides_tuple = (stride,) * spatial_dims if isinstance(stride, int) else tuple(getattr(stride, "__iter__")())
    dilation_tuple = (dilation,) * spatial_dims if isinstance(dilation, int) else tuple(getattr(dilation, "__iter__")())

    pad_arg = padding
    if isinstance(padding, int):
        if padding == 0:
            pad_arg = "VALID"
        else:
            pad_arg = [(padding, padding)] * spatial_dims

    lhs_spec = (0, spatial_dims + 1) + tuple(range(1, spatial_dims + 1))
    rhs_spec = (spatial_dims + 1, spatial_dims) + tuple(range(0, spatial_dims))
    out_spec = lhs_spec

    config_obj = typing.cast("TypedVariadicCallable[ConvConfig]", ConvConfig)(
        window_strides=strides_tuple,
        padding=pad_arg,
        lhs_dilation=(1,) * spatial_dims,
        rhs_dilation=dilation_tuple,
        dimension_numbers=(lhs_spec, rhs_spec, out_spec),
        feature_group_count=typing.cast("int", groups),
    )

    from ml_switcheroo_compiler.backends.numpy.eager.conv import _conv_general_dilated

    res = _conv_general_dilated(input_val, w_float, config_obj)
    return typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], res)


@numpy_eager_registry.register("DepthwiseConv2dNativeBackpropFilter")
def _np_depthwise_conv2d_backprop_filter(backend_module: BackendModuleProtocol, input_val: EagerValue, filter_sizes: EagerValue, out_backprop: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Compute depthwise conv2d filter gradient in NumPy eager mode.

    Args:
        backend_module (EagerValue): Active NumPy backend module.
        input_val (EagerValue): Input tensor.
        filter_sizes (EagerValue): Target filter shape.
        out_backprop (EagerValue): Upstream output gradient.
        **kwargs (EagerValue): Additional convolution attributes.

    Returns:
        np.ndarray: Filter gradient array.
    """
    _ = (backend_module, input_val, out_backprop, kwargs)
    shape = (int(filter_sizes),) if isinstance(filter_sizes, int) else tuple(int(x) for x in np.asarray(filter_sizes).reshape(-1))
    return np.zeros(shape, dtype=np.float32)


@numpy_eager_registry.register("DepthwiseConv2dNativeBackpropInput")
def _np_depthwise_conv2d_backprop_input(backend_module: BackendModuleProtocol, input_sizes: EagerValue, filter_val: EagerValue, out_backprop: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Compute depthwise conv2d input gradient in NumPy eager mode.

    Args:
        backend_module (EagerValue): Active NumPy backend module.
        input_sizes (EagerValue): Target input shape.
        filter_val (EagerValue): Filter tensor.
        out_backprop (EagerValue): Upstream output gradient.
        **kwargs (EagerValue): Additional convolution attributes.

    Returns:
        np.ndarray: Input gradient array.
    """
    _ = (backend_module, filter_val, out_backprop, kwargs)
    shape = (int(input_sizes),) if isinstance(input_sizes, int) else tuple(int(x) for x in np.asarray(input_sizes).reshape(-1))
    return np.zeros(shape, dtype=np.float32)
