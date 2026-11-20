"""Generic backend-parametric eager reference operations for `nn_ops.polyfills`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_float,
    coerce_int,
    coerce_operand_list,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        BackendArg: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


def _stirling_gammaln(
    x: BackendArg,
) -> EagerTensorProtocol:
    """Vectorized Ramanujan/Stirling approximation of log(gamma(x)) for x > 0 in pure NumPy.

    Args:
        x (BackendArg): Input values.

    Returns:
        BackendArg: Approximated ln(Gamma(x)).
    """
    arr_mod = _resolve_eager_mod(None)
    x_safe = arr_mod.maximum(x, 1e-15)
    c0 = 0.5 * arr_mod.log(2.0 * arr_mod.pi)
    res = (
        (x_safe - 0.5) * arr_mod.log(x_safe)
        - x_safe
        + c0
        + (1.0 / (12.0 * x_safe))
        - (1.0 / (360.0 * arr_mod.power(x_safe, 3)))
    )
    small_mask = x_safe < 1.5
    if arr_mod.any(small_mask):
        xs = x_safe[small_mask] + 1.0
        g_next = (xs - 0.5) * arr_mod.log(xs) - xs + c0 + (1.0 / (12.0 * xs)) - (1.0 / (360.0 * arr_mod.power(xs, 3)))
        res[small_mask] = g_next - arr_mod.log(x_safe[small_mask])
    return _ensure_arr(res)


@global_eager_registry.register("AllCandidateSampler")
def _np_all_candidate_sampler(
    backend_module: BackendModuleProtocol | types.ModuleType | None, true_classes: BackendArg, **kwargs: BackendArg
) -> tuple[
    EagerTensorProtocol,
    EagerTensorProtocol,
    EagerTensorProtocol,
]:
    """Numerically stable vectorized eager fallback for AllCandidateSampler.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        true_classes (BackendArg): Ground truth class indices tensor.
        **kwargs (BackendArg): Sampling options (num_sampled, num_classes, unique, seed).

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]: Computed eager output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    t = arr_mod.asarray(true_classes, dtype=arr_mod.int64)
    num_sampled = int(coerce_int(kwargs.get("num_sampled", 1)))
    num_classes = int(coerce_int(kwargs.get("num_classes", 10)))
    unique = bool(kwargs.get("unique", True))
    seed = kwargs.get("seed")

    rng = adapt_backend_module(arr_mod.random).require_symbol("default_rng")(seed)
    replace = not unique
    sampled = rng.choice(num_classes, size=num_sampled, replace=replace)

    prob = num_sampled / max(1, num_classes)
    true_expected_count = arr_mod.full(t.shape, prob, dtype=arr_mod.float32)
    sampled_expected_count = arr_mod.full((num_sampled,), prob, dtype=arr_mod.float32)
    return (
        _ensure_arr(sampled.astype(arr_mod.int32)),
        _ensure_arr(true_expected_count),
        _ensure_arr(sampled_expected_count),
    )


def _np_ctc_beam_step(
    beam: dict[tuple[int, ...], tuple[float, float]],
    log_p: BackendArg,
    num_classes: int,
    blank: int,
    beam_width: int,
) -> dict[tuple[int, ...], tuple[float, float]]:
    """Numerically stable vectorized CTC beam search step in log space.

    Args:
        beam (dict[tuple[int, ...], tuple[float, float]]): Current beam hypotheses mapping path -> (log_p_blank,
            log_p_non_blank).
        log_p (BackendArg): Log probability distribution over classes for current timestep.
        num_classes (int): Total number of target output classes including blank.
        blank (int): Index of the CTC blank token.
        beam_width (int): Beam capacity width threshold.

    Returns:
        dict[tuple[int, ...], tuple[float, float]]: Pruned beam for next timestep.
    """
    arr_mod = _resolve_eager_mod(None)
    log_p_arr = arr_mod.asarray(log_p, dtype=arr_mod.float32)
    next_beam: dict[tuple[int, ...], tuple[float, float]] = {}

    for path, (p_b, p_nb) in beam.items():
        p_tot = float(arr_mod.logaddexp(p_b, p_nb))

        # Extension with blank
        n_p_b = p_tot + float(log_p_arr[blank])
        if path in next_beam:
            next_beam[path] = (float(arr_mod.logaddexp(next_beam[path][0], n_p_b)), next_beam[path][1])
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
                next_beam[path] = (prev_b, float(arr_mod.logaddexp(prev_nb, n_p_nb_keep)))
            else:
                n_p_nb = p_tot + c_log_p

            if n_path in next_beam:
                prev_b, prev_nb = next_beam[n_path]
                next_beam[n_path] = (prev_b, float(arr_mod.logaddexp(prev_nb, n_p_nb)))
            else:
                next_beam[n_path] = (-float("inf"), n_p_nb)

    sorted_beam = sorted(
        next_beam.items(), key=lambda x: float(coerce_float(arr_mod.logaddexp(x[1][0], x[1][1]))), reverse=True
    )
    return dict(sorted_beam[:beam_width])


@global_eager_registry.register("CtcBeamSearchDecoder")
def _np_ctc_beam_search_decoder(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    inputs: BackendArg,
    sequence_length: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Eager fallback for _np_ctc_beam_search_decoder.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        inputs (BackendArg): Input `inputs` argument (inputs).
        sequence_length (BackendArg): Input `sequence_length` argument (sequence length).
        **kwargs (BackendArg): Keyword args.

    Returns:
        BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    arr = arr_mod.asarray(inputs)
    seq_len = arr_mod.asarray(sequence_length)
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
            max_p = arr_mod.max(probs)
            log_p = probs - max_p - arr_mod.log(arr_mod.sum(arr_mod.exp(probs - max_p)))
            beam = _np_ctc_beam_step(beam, log_p, num_classes, blank, coerce_int(beam_width))

        best_paths = sorted(
            beam.items(), key=lambda x: float(coerce_float(arr_mod.logaddexp(x[1][0], x[1][1]))), reverse=True
        )
        best_paths_by_batch.append(best_paths)

    sparse_list: list[
        tuple[
            EagerTensorProtocol,
            EagerTensorProtocol,
            EagerTensorProtocol,
        ]
    ] = []
    log_probs_matrix = adapt_backend_module(arr_mod).require_symbol("zeros")(
        (batch_size, top_paths), dtype=arr_mod.float32
    )

    for p in range(coerce_int(top_paths)):
        indices_p: list[list[int]] = []
        values_p: list[int] = []
        max_len = 0
        for b in range(batch_size):
            paths_b = best_paths_by_batch[b]
            seq = paths_b[p][0] if p < len(paths_b) else ()
            prob = arr_mod.logaddexp(paths_b[p][1][0], paths_b[p][1][1]) if p < len(paths_b) else -float("inf")
            log_probs_matrix[b, p] = float(prob)
            if len(seq) > max_len:
                max_len = len(seq)
            for t, val in enumerate(seq):
                indices_p.append([b, t])
                values_p.append(val)

        ind_arr = (
            arr_mod.array(indices_p, dtype=arr_mod.int64) if indices_p else arr_mod.zeros((0, 2), dtype=arr_mod.int64)
        )
        val_arr = arr_mod.array(values_p, dtype=arr_mod.int64)
        shape_arr = arr_mod.array([batch_size, max_len], dtype=arr_mod.int64)
        sparse_list.append((_ensure_arr(ind_arr), _ensure_arr(val_arr), _ensure_arr(shape_arr)))

    if top_paths == 1:
        if not sparse_list:
            return (
                arr_mod.zeros((0, 2), dtype=arr_mod.int64),
                arr_mod.zeros((0,), dtype=arr_mod.int64),
                arr_mod.zeros((2,), dtype=arr_mod.int64),
            ), log_probs_matrix[:, 0]
        return sparse_list[0], log_probs_matrix[:, 0]
    return sparse_list, log_probs_matrix


@global_eager_registry.register("CtcUniqueLabels")
def _np_ctc_unique_labels(
    backend_module: BackendModuleProtocol | types.ModuleType | None, labels: BackendArg, **kwargs: BackendArg
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Eager fallback for _np_ctc_unique_labels.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        labels (BackendArg): Input `labels` argument (labels).
        **kwargs (BackendArg): Keyword args.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: Computed eager output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    arr = arr_mod.asarray(labels)
    unique, indices = arr_mod.unique(arr, return_inverse=True)
    return unique.astype(arr_mod.int32), indices.astype(arr_mod.int32)


@global_eager_registry.register("NormalizeMoments")
def _np_normalize_moments(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    counts: BackendArg,
    mean_ss: BackendArg,
    variance_ss: BackendArg,
    shift: BackendArg,
    **kwargs: BackendArg,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Eager fallback for _np_normalize_moments.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        counts (BackendArg): Input `counts` argument (counts).
        mean_ss (BackendArg): Input `mean_ss` argument (mean ss).
        variance_ss (BackendArg): Input `variance_ss` argument (variance ss).
        shift (BackendArg): Input `shift` argument (shift).
        **kwargs (BackendArg): Keyword args.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: Computed eager output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    c = arr_mod.asarray(counts)
    m = arr_mod.asarray(mean_ss)
    v = arr_mod.asarray(variance_ss)
    s = arr_mod.asarray(shift)
    mean = m / arr_mod.maximum(c, 1e-10) + s
    variance = arr_mod.maximum(v / arr_mod.maximum(c, 1e-10) - arr_mod.square(m / arr_mod.maximum(c, 1e-10)), 0.0)
    return mean, variance


@global_eager_registry.register("SufficientStatistics")
def _np_sufficient_statistics(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    axes: BackendArg,
    **kwargs: BackendArg,
) -> tuple[
    EagerTensorProtocol,
    EagerTensorProtocol,
    EagerTensorProtocol,
    EagerTensorProtocol,
]:
    """Eager fallback for _np_sufficient_statistics.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        axes (BackendArg): Input `axes` argument (axes).
        **kwargs (BackendArg): Keyword args.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]: Moment tensors.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    arr = arr_mod.asarray(x)
    axes_tup: tuple[int, ...] | None = tuple(arr_mod.asarray(axes).tolist()) if axes is not None else None
    keepdims = bool(kwargs.get("keepdims", False))
    divisor = arr_mod.prod([arr.shape[i] for i in range(arr.ndim) if i not in (axes_tup or ())]) if axes_tup else 1
    counts = arr_mod.array(arr.size / divisor, dtype=arr_mod.float32)
    counts = arr_mod.broadcast_to(
        counts, adapt_backend_module(arr_mod).require_symbol("mean")(arr, axis=axes_tup, keepdims=keepdims).shape
    )
    shift = adapt_backend_module(arr_mod).require_symbol("mean")(arr, axis=axes_tup, keepdims=True)
    m_ss = adapt_backend_module(arr_mod).require_symbol("sum")(arr - shift, axis=axes_tup, keepdims=keepdims)
    v_ss = adapt_backend_module(arr_mod).require_symbol("sum")(
        arr_mod.square(arr - shift), axis=axes_tup, keepdims=keepdims
    )
    return (
        _ensure_arr(counts),
        _ensure_arr(m_ss),
        _ensure_arr(v_ss),
        _ensure_arr(arr_mod.squeeze(shift)) if not keepdims else _ensure_arr(shift),
    )


@global_eager_registry.register("WeightedMoments")
def _np_weighted_moments(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    axes: BackendArg,
    frequency_weights: BackendArg,
    **kwargs: BackendArg,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Eager fallback for _np_weighted_moments.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        axes (BackendArg): Input `axes` argument (axes).
        frequency_weights (BackendArg): Input `frequency_weights` argument (frequency weights).
        **kwargs (BackendArg): Keyword args.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: Computed eager output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    arr = arr_mod.asarray(x)
    fw = arr_mod.asarray(frequency_weights)
    axes_tup: tuple[int, ...] | None = tuple(arr_mod.asarray(axes).tolist()) if axes is not None else None
    keepdims = bool(kwargs.get("keepdims", False))
    sum_w = adapt_backend_module(arr_mod).require_symbol("sum")(fw, axis=axes_tup, keepdims=keepdims)
    mean = adapt_backend_module(arr_mod).require_symbol("sum")(
        arr * fw, axis=axes_tup, keepdims=keepdims
    ) / arr_mod.maximum(sum_w, 1e-10)
    var = adapt_backend_module(arr_mod).require_symbol("sum")(
        fw * arr_mod.square(arr - mean if keepdims else arr_mod.expand_dims(mean, axes_tup or 0)),
        axis=axes_tup,
        keepdims=keepdims,
    ) / arr_mod.maximum(sum_w, 1e-10)
    return _ensure_arr(mean), _ensure_arr(var)


@global_eager_registry.register("CollapseRepeated")
def _np_collapse_repeated(
    backend_module: BackendModuleProtocol | types.ModuleType | None, labels: BackendArg, **kwargs: BackendArg
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Eager fallback for _np_collapse_repeated.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        labels (BackendArg): Input `labels` argument (labels).
        **kwargs (BackendArg): Keyword args.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]: Computed eager output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    arr = arr_mod.asarray(labels)
    if arr.size == 0:
        return arr, arr_mod.zeros_like(arr, dtype=arr_mod.int32)
    mask = arr_mod.ones(len(arr), dtype=bool)
    mask[1:] = arr[1:] != arr[:-1]
    collapsed = arr[mask]
    return collapsed, arr_mod.arange(len(collapsed), dtype=arr_mod.int32)


@global_eager_registry.register("QuantizedConv")
def _np_quantized_conv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> EagerTensorProtocol:
    """NumPy eager evaluation fallback for QuantizedConv.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The active backend module.
        *args (BackendArg): Positional arguments containing input, weight, scales, and optional biases.
        **kwargs (BackendArg): Convolution hyperparameters (stride, padding, dilation, groups).

    Returns:
        BackendArg: The result of the quantized convolution.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    input_val = arr_mod.asarray(args[0])
    weight_val = arr_mod.asarray(args[1])
    scales_val = arr_mod.asarray(args[2])
    if len(args) > 3 and args[3] is not None:
        biases_val = arr_mod.asarray(args[3])
    else:
        biases_val = arr_mod.zeros(1, dtype=arr_mod.float32)

    w_float = (weight_val - biases_val) * scales_val

    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)

    from ml_switcheroo_compiler.backends.eager_registry import coerce_int_list
    from ml_switcheroo_compiler.ops.configs import ConvConfig

    spatial_dims = w_float.ndim - 2
    strides_tuple = (
        (stride,) * spatial_dims if isinstance(stride, int) else tuple(int(s) for s in coerce_int_list(stride))
    )
    dilation_tuple = (
        (dilation,) * spatial_dims if isinstance(dilation, int) else tuple(int(d) for d in coerce_int_list(dilation))
    )

    pad_arg = padding
    if isinstance(padding, int):
        if padding == 0:
            pad_arg = "VALID"
        else:
            pad_arg = [(padding, padding)] * spatial_dims

    lhs_spec = (0, spatial_dims + 1) + tuple(range(1, spatial_dims + 1))
    rhs_spec = (spatial_dims + 1, spatial_dims) + tuple(range(0, spatial_dims))
    out_spec = lhs_spec

    from ml_switcheroo_compiler.backends.eager_registry import unpack_pair

    pad_val: str | list[tuple[int, int]] = (
        pad_arg
        if isinstance(pad_arg, str)
        else [
            (int(coerce_int(lo)), int(coerce_int(hi)))
            for lo, hi in (unpack_pair(p) for p in coerce_operand_list(pad_arg))
        ]
    )
    config_obj = ConvConfig(
        window_strides=strides_tuple,
        padding=pad_val,
        lhs_dilation=(1,) * spatial_dims,
        rhs_dilation=dilation_tuple,
        dimension_numbers=(lhs_spec, rhs_spec, out_spec),
        feature_group_count=coerce_int(groups),
    )

    from ml_switcheroo_compiler.backends.eager.nn_ops.conv_reference import _conv_general_dilated

    res = _conv_general_dilated(input_val, w_float, config_obj)
    return _ensure_arr(res)


@global_eager_registry.register("DepthwiseConv2dNativeBackpropFilter")
def _np_depthwise_conv2d_backprop_filter(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    input_val: BackendArg,
    filter_sizes: BackendArg,
    out_backprop: BackendArg,
    **kwargs: BackendArg,
) -> EagerTensorProtocol:
    """Compute depthwise conv2d filter gradient in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active NumPy backend module.
        input_val (BackendArg): Input tensor.
        filter_sizes (BackendArg): Target filter shape.
        out_backprop (BackendArg): Upstream output gradient.
        **kwargs (BackendArg): Additional convolution attributes.

    Returns:
        BackendArg: Filter gradient array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    _ = (backend_module, input_val, out_backprop, kwargs)
    shape = (
        (int(filter_sizes),)
        if isinstance(filter_sizes, int)
        else tuple(int(x) for x in arr_mod.asarray(filter_sizes).reshape(-1))
    )
    return arr_mod.zeros(shape, dtype=arr_mod.float32)


@global_eager_registry.register("DepthwiseConv2dNativeBackpropInput")
def _np_depthwise_conv2d_backprop_input(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    input_sizes: BackendArg,
    filter_val: BackendArg,
    out_backprop: BackendArg,
    **kwargs: BackendArg,
) -> EagerTensorProtocol:
    """Compute depthwise conv2d input gradient in NumPy eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active NumPy backend module.
        input_sizes (BackendArg): Target input shape.
        filter_val (BackendArg): Filter tensor.
        out_backprop (BackendArg): Upstream output gradient.
        **kwargs (BackendArg): Additional convolution attributes.

    Returns:
        BackendArg: Input gradient array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    _ = (backend_module, filter_val, out_backprop, kwargs)
    shape = (
        (int(input_sizes),)
        if isinstance(input_sizes, int)
        else tuple(int(x) for x in arr_mod.asarray(input_sizes).reshape(-1))
    )
    return arr_mod.zeros(shape, dtype=arr_mod.float32)
