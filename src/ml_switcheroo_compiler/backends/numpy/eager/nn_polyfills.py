# ruff: noqa: C901, PLR0912, F841, PLR0917
"""Numpy eager fallback implementations for stubbed NN operations."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("IsotonicRegression")
def _np_isotonic_regression(backend_module: object, y: object, **kwargs: object) -> tuple:
    y_arr = np.asarray(y, dtype=np.float32)
    res = np.copy(y_arr)
    w = np.ones_like(y_arr)
    n = len(y_arr)

    blocks = [[i] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        v1 = np.sum(res[blocks[i]] * w[blocks[i]]) / np.sum(w[blocks[i]])
        v2 = np.sum(res[blocks[i + 1]] * w[blocks[i + 1]]) / np.sum(w[blocks[i + 1]])
        if v1 > v2:
            blocks[i] = blocks[i] + blocks[i + 1]
            del blocks[i + 1]
            new_val = np.sum(res[blocks[i]] * w[blocks[i]]) / np.sum(w[blocks[i]])
            for idx in blocks[i]:
                res[idx] = new_val
            i = max(0, i - 1)
        else:
            i += 1

    return res, np.zeros_like(res, dtype=np.int32)


@numpy_eager_registry.register("ConvTranspose")
def _np_conv_transpose(backend_module: object, input: object, filters: object, **kwargs: object) -> object:
    import scipy.signal

    arr = np.asarray(input)
    f = np.asarray(filters)
    out_shape = kwargs.get("output_shape", None)

    spatial_dims = arr.ndim - 2
    if spatial_dims not in (1, 2, 3):
        return np.zeros(out_shape if out_shape else arr.shape, dtype=arr.dtype)

    if out_shape is None:
        out_shape_list = [arr.shape[0]]
        for i in range(spatial_dims):
            out_shape_list.append(arr.shape[i + 1] + f.shape[i] - 1)
        out_shape_list.append(f.shape[-1])
        out_shape = tuple(out_shape_list)

    out = np.zeros(out_shape, dtype=arr.dtype)
    for c_out in range(out_shape[-1]):
        for c_in in range(arr.shape[-1]):
            f_expanded = f[..., c_in, c_out]
            for _ in range(arr.ndim - f_expanded.ndim - 1):
                f_expanded = np.expand_dims(f_expanded, 0)
            res = scipy.signal.convolve(arr[..., c_in], f_expanded, mode="full")
            slices = (slice(None),) + tuple(slice(0, out_shape[i + 1]) for i in range(spatial_dims))
            out[..., c_out] += res[slices]
    return out


@numpy_eager_registry.register("DepthwiseConv2dBackpropFilter")
def _np_depthwise_conv2d_backprop_filter(backend_module: object, input: object, filter_sizes: object, out_backprop: object, **kwargs: object) -> object:
    import scipy.signal

    arr = np.asarray(input)
    ob = np.asarray(out_backprop)
    f_shape = tuple(filter_sizes)
    out = np.zeros(f_shape, dtype=arr.dtype)
    if arr.ndim == 4 and ob.ndim == 4:
        for c in range(f_shape[2]):
            for m in range(f_shape[3]):
                res = scipy.signal.correlate(arr[..., c], ob[..., c * f_shape[3] + m], mode="valid")
                out[..., c, m] += res[0]
    return out


@numpy_eager_registry.register("DepthwiseConv2dBackpropInput")
def _np_depthwise_conv2d_backprop_input(backend_module: object, input_sizes: object, filter: object, out_backprop: object, **kwargs: object) -> object:
    import scipy.signal

    f = np.asarray(filter)
    ob = np.asarray(out_backprop)
    input_sizes = tuple(input_sizes)
    out = np.zeros(input_sizes, dtype=f.dtype)
    if out.ndim == 4 and ob.ndim == 4:
        for c in range(f.shape[2]):
            for m in range(f.shape[3]):
                f_expanded = np.expand_dims(f[..., c, m], 0)
                res = scipy.signal.convolve(ob[..., c * f.shape[3] + m], f_expanded, mode="full")
                out[..., c] += res[:, : out.shape[1], : out.shape[2]]
    return out


@numpy_eager_registry.register("Dilation2d")
def _np_dilation2d(backend_module: object, input: object, filter: object, **kwargs: object) -> object:
    import scipy.ndimage

    arr = np.asarray(input)
    f = np.asarray(filter)
    if arr.ndim == 4:
        out = np.zeros_like(arr)
        for c in range(arr.shape[-1]):
            struct = np.expand_dims(f[..., c], 0)
            out[..., c] = scipy.ndimage.grey_dilation(arr[..., c], structure=struct)
        return out
    return scipy.ndimage.grey_dilation(arr, structure=f)


@numpy_eager_registry.register("Erosion2d")
def _np_erosion2d(backend_module: object, value: object, kernel: object, **kwargs: object) -> object:
    import scipy.ndimage

    arr = np.asarray(value)
    f = np.asarray(kernel)
    if arr.ndim == 4:
        out = np.zeros_like(arr)
        for c in range(arr.shape[-1]):
            struct = np.expand_dims(f[..., c], 0)
            out[..., c] = scipy.ndimage.grey_erosion(arr[..., c], structure=struct)
        return out
    return scipy.ndimage.grey_erosion(arr, structure=f)


@numpy_eager_registry.register("InTopK")
def _np_in_top_k(backend_module: object, targets: object, predictions: object, **kwargs: object) -> object:
    t = np.asarray(targets)
    p = np.asarray(predictions)
    k = kwargs.get("k", 1)
    if p.ndim == 1:
        top_k_indices = np.argsort(p)[-k:]
        return np.isin(t, top_k_indices)
    top_k_indices = np.argsort(p, axis=-1)[:, -k:]
    return np.any(top_k_indices == t[..., None], axis=-1)


@numpy_eager_registry.register("LogPoissonLoss")
def _np_log_poisson_loss(backend_module: object, targets: object, log_input: object, **kwargs: object) -> object:
    t = np.asarray(targets)
    log_inp = np.asarray(log_input)
    compute_full = kwargs.get("compute_full_loss", False)
    res = np.exp(log_inp) - t * log_inp
    if compute_full:
        import scipy.special

        res += scipy.special.gammaln(t + 1)
    return res


@numpy_eager_registry.register("AllCandidateSampler")
def _np_all_candidate_sampler(backend_module: object, true_classes: object, **kwargs: object) -> tuple:
    t = np.asarray(true_classes)
    num_sampled = kwargs.get("num_sampled", 1)
    num_classes = kwargs.get("num_classes", 10)
    sampled = np.random.choice(num_classes, num_sampled, replace=True)
    true_expected_count = np.ones_like(t, dtype=np.float32) * (num_sampled / num_classes)
    sampled_expected_count = np.ones((num_sampled,), dtype=np.float32) * (num_sampled / num_classes)
    return sampled.astype(np.int32), true_expected_count, sampled_expected_count


@numpy_eager_registry.register("CtcBeamSearchDecoder")
def _np_ctc_beam_search_decoder(backend_module: object, inputs: object, sequence_length: object, **kwargs: object) -> tuple:
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
        beam = {(): (0.0, -float("inf"))}

        for t in range(T):
            probs = arr[t, b]
            # Convert to log probs if they aren't already. Assuming they are logits.
            # But the TF spec says inputs are logits.
            max_p = np.max(probs)
            log_p = probs - max_p - np.log(np.sum(np.exp(probs - max_p)))

            next_beam = {}

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
            beam = dict(sorted_beam[:beam_width])

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
        indices = np.zeros((0, 2), dtype=np.int64)

    sparse = (np.array(indices, dtype=np.int64), np.array(values, dtype=np.int64), np.array([batch_size, max((len(s) for s in decoded), default=0)], dtype=np.int64))

    # Usually it returns a list of sparse tensors if top_paths > 1, but we return a tuple containing the sparse tensor
    # depending on TF spec. We just return a tuple (sparse_tensor, log_probs)
    return sparse, np.array(log_probs, dtype=np.float32)


@numpy_eager_registry.register("CtcGreedyDecoder")
def _np_ctc_greedy_decoder(backend_module: object, inputs: object, sequence_length: object, **kwargs: object) -> tuple:
    arr = np.asarray(inputs)
    argmax = np.argmax(arr, axis=-1)
    decoded = []
    for b in range(argmax.shape[1]):
        seq = argmax[: sequence_length[b], b]
        if len(seq) > 0:
            mask = np.ones(len(seq), dtype=bool)
            mask[1:] = seq[1:] != seq[:-1]
            seq = seq[mask]
        decoded.append(seq[seq != (arr.shape[-1] - 1)])
    indices = []
    values = []
    for b, seq in enumerate(decoded):
        for t, val in enumerate(seq):
            indices.append([b, t])
            values.append(val)
    if len(indices) == 0:
        indices = np.zeros((0, 2), dtype=np.int64)
    sparse = (np.array(indices, dtype=np.int64), np.array(values, dtype=np.int64), np.array([argmax.shape[1], max((len(s) for s in decoded), default=0)], dtype=np.int64))
    return sparse, np.zeros((argmax.shape[1],), dtype=np.float32)


@numpy_eager_registry.register("CtcUniqueLabels")
def _np_ctc_unique_labels(backend_module: object, labels: object, **kwargs: object) -> tuple:
    arr = np.asarray(labels)
    unique, indices = np.unique(arr, return_inverse=True)
    return unique.astype(np.int32), indices.astype(np.int32)


@numpy_eager_registry.register("NormalizeMoments")
def _np_normalize_moments(backend_module: object, counts: object, mean_ss: object, variance_ss: object, shift: object, **kwargs: object) -> tuple:
    c = np.asarray(counts)
    m = np.asarray(mean_ss)
    v = np.asarray(variance_ss)
    s = np.asarray(shift)
    mean = m / np.maximum(c, 1e-10) + s
    variance = np.maximum(v / np.maximum(c, 1e-10) - np.square(m / np.maximum(c, 1e-10)), 0.0)
    return mean, variance


@numpy_eager_registry.register("SufficientStatistics")
def _np_sufficient_statistics(backend_module: object, x: object, axes: object, **kwargs: object) -> tuple:
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
def _np_weighted_moments(backend_module: object, x: object, axes: object, frequency_weights: object, **kwargs: object) -> tuple:
    arr = np.asarray(x)
    fw = np.asarray(frequency_weights)
    axes = tuple(np.asarray(axes).tolist()) if axes is not None else None
    keepdims = kwargs.get("keepdims", False)
    sum_w = np.sum(fw, axis=axes, keepdims=keepdims)
    mean = np.sum(arr * fw, axis=axes, keepdims=keepdims) / np.maximum(sum_w, 1e-10)
    var = np.sum(fw * np.square(arr - mean if keepdims else np.expand_dims(mean, axes)), axis=axes, keepdims=keepdims) / np.maximum(sum_w, 1e-10)
    return mean, var


@numpy_eager_registry.register("MaxPoolWithArgmax")
def _np_max_pool_with_argmax(backend_module: object, input: object, **kwargs: object) -> tuple:
    arr = np.asarray(input)
    import scipy.ndimage

    size = kwargs.get("pool_size", (2, 2))
    if isinstance(size, int):
        size = (1, size, size, 1) if arr.ndim == 4 else (size, size)
    maxes = scipy.ndimage.maximum_filter(arr, size=size)
    argmaxes = np.argmax(arr.reshape(arr.shape[0], -1), axis=1)
    return maxes, argmaxes.astype(np.int32)


@numpy_eager_registry.register("CollapseRepeated")
def _np_collapse_repeated(backend_module: object, labels: object, **kwargs: object) -> tuple:
    arr = np.asarray(labels)
    if arr.size == 0:
        return arr, np.zeros_like(arr, dtype=np.int32)
    mask = np.ones(len(arr), dtype=bool)
    mask[1:] = arr[1:] != arr[:-1]
    collapsed = arr[mask]
    return collapsed, np.arange(len(collapsed), dtype=np.int32)


@numpy_eager_registry.register("QuantizedConv")
def _np_quantized_conv(backend_module: object, *args: object, **kwargs: object) -> object:
    """NumPy eager evaluation fallback for QuantizedConv.

    Args:
        backend_module (object): The active backend module.
        *args: Positional arguments containing input, weight, scales, and optional biases.
        **kwargs: Convolution hyperparameters (stride, padding, dilation, groups).

    Returns:
        object: The result of the quantized convolution.
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

    pad_arg: object = padding
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
