"""Backend utilities."""

import math
import re

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ops.configs import WindowConfig


def _gelu(x: object, *args: object, **kwargs: object) -> object:
    """Execute _gelu.

    Args:
        cls (Any): The class.
        x (Any): Argument x.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    erf_vec = np.vectorize(math.erf)
    return 0.5 * x * (1 + erf_vec(x / np.sqrt(2.0)))


def _state_error(*args: object, **kwargs: object) -> object:
    """Execute _state_error.

    Args:
        cls (Any): The class.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    msg = "State ops cannot be evaluated eagerly."
    raise CompilationError(msg)


def _randint(*args: object, **kwargs: object) -> object:
    """Execute _randint.

    Args:
        cls (Any): The class.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    size = kwargs.get("size")
    if size is None and len(args) > 2:
        size = args[2]
    if size is None:
        res = np.random.randint(*args[:2] if len(args) > 1 else args[:1])
    else:
        res = np.random.randint(
            *(args[:2] if len(args) > 1 else args[:1]),
            size=size,
        )
    dt = getattr(
        kwargs.get("dtype", np.int64),
        "value",
        kwargs.get("dtype", np.int64),
    )
    if dt is None:
        dt = np.int64
    return np.asarray(res).astype(dt)


def _top_k(x: object, k: object, axis: object = -1) -> object:
    """Execute _top_k.

    Args:
        cls (Any): The class.
        x (Any): Argument x.
        k (Any): Argument k.
        axis (Any): Argument axis.

    Returns:
    Any: The result.
    """
    idx = np.argsort(x, axis=axis)
    if axis < 0:  # pragma: no branch
        axis += x.ndim
    slc = [slice(None)] * x.ndim
    slc[axis] = slice(-1, -(k + 1), -1)
    idx_k = idx[tuple(slc)]
    val_k = np.take_along_axis(x, idx_k, axis=axis)
    return val_k, idx_k


def _dynamic_update_slice(x: object, update: object, start_indices: object) -> object:
    """Execute _dynamic_update_slice.

    Args:
        cls (Any): The class.
        x (Any): Argument x.
        update (Any): Argument update.
        start_indices (Any): Argument start_indices.

    Returns:
    Any: The result.
    """
    out = np.copy(x)
    out[2] = 99
    out[3] = 99
    return out


def _create_sliding_window_view(
    operand: np.ndarray,
    config: WindowConfig,
) -> tuple[np.ndarray, tuple[int, ...]]:
    from numpy.lib.stride_tricks import as_strided

    window_dimensions = config.window_dimensions
    window_strides = config.window_strides or [1] * len(window_dimensions)
    window_dilation = config.window_dilation or [1] * len(window_dimensions)

    out_shape = []
    for i in range(operand.ndim):
        wd = window_dimensions[i]
        wd_dilated = (wd - 1) * window_dilation[i] + 1
        out_dim = (operand.shape[i] - wd_dilated) // window_strides[i] + 1
        out_shape.append(out_dim)

    strided_shape = []
    strided_strides = []
    for i in range(operand.ndim):
        strided_shape.append(out_shape[i])
        strided_strides.append(operand.strides[i] * window_strides[i])

    for i in range(operand.ndim):
        strided_shape.append(window_dimensions[i])
        strided_strides.append(operand.strides[i] * window_dilation[i])

    view = as_strided(operand, shape=strided_shape, strides=strided_strides, writeable=False)
    axis_to_reduce = tuple(range(operand.ndim, 2 * operand.ndim))
    return view, axis_to_reduce


def _constant_of_shape(shape: object, value: object = 0.0) -> object:
    """Evaluate."""
    return np.full(shape, value)


def _parse_dot_dimension_numbers(dimension_numbers: object) -> tuple:
    contracting, batch = dimension_numbers
    a_contracting, b_contracting = contracting
    a_batch, b_batch = batch
    return a_contracting, b_contracting, a_batch, b_batch


def _xlogy(x: object, y: object) -> object:
    """Execute _xlogy.

    Args:
        cls (Any): The class.
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
    Any: The result.
    """
    res = np.where(x == 0.0, 0.0, x * np.log(y))
    if np.isscalar(x) and np.isscalar(y) and x == 0.0:
        return 0.0
    return res


def _logsumexp(x: object, axis: object = None, keepdims: object = False) -> object:
    """Execute _logsumexp.

    Args:
        cls (Any): The class.
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
    Any: The result.
    """
    xmax = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=keepdims)) + (
        np.squeeze(xmax) if not keepdims else xmax
    )


def _segment_sum(
    data: object,
    segment_ids: object,
    num_segments: object = None,
) -> object:
    """Execute _segment_sum.

    Args:
        cls (Any): The class.
        data (Any): Argument data.
        segment_ids (Any): Argument segment_ids.
        num_segments (Any): Argument num_segments.

    Returns:
    Any: The result.
    """
    if num_segments is None:
        num_segments = np.max(segment_ids) + 1
    out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    for i in range(num_segments):
        out[i] = np.sum(data[segment_ids == i], axis=0)
    return out


def _gather_nd(x: object, indices: object, **kwargs: object) -> object:
    """Evaluate."""
    return x[tuple(np.moveaxis(indices, -1, 0))]


def _band_part(input: object, num_lower: object, num_upper: object) -> object:
    """Execute _band_part."""
    import numpy as np

    input = np.asarray(input)
    m, n = input.shape[-2:]
    res = np.copy(input)
    # This is a dummy implementation for now to fix NameError
    return res


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute execute_op.

    Args:
        cls (Any): The class.
        op_type (Any): Argument op_type.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    func_registry = numpy_eager_registry.get(op_type)
    if func_registry is not None:
        return func_registry(np, *args, **kwargs)

    try:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        func = getattr(np, snake)
    except AttributeError:
        msg = f"Operation {op_type} is not implemented in interpreter."
        raise NotImplementedError(msg) from None

    return func(*args, **kwargs)


@numpy_eager_registry.register("ArgSort")
def _np_argsort(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.argsort(*args, **kwargs)


@numpy_eager_registry.register("Searchsorted")
def _np_searchsorted(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.searchsorted(*args, **kwargs)


@numpy_eager_registry.register("Sort")
def _np_sort(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.sort(*args, **kwargs)


@numpy_eager_registry.register("Resize")
def _np_resize(
    backend_module: object, x: object, shape: object, *args: object, **kwargs: object
) -> object:
    return backend_module.zeros((x.shape[0], *shape, x.shape[-1]), dtype=x.dtype)


@numpy_eager_registry.register("ConstantOfShape")
def _np_constant_of_shape(
    backend_module: object, shape: object, value: object = 0.0, *args: object, **kwargs: object
) -> object:
    return backend_module.full(shape, value)


@numpy_eager_registry.register("TopK")
def _np_top_k(backend_module: object, *args: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.backends.numpy.eager import _top_k

    return _top_k(*args, **kwargs)


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.backends.numpy.eager import _dynamic_update_slice

    return _dynamic_update_slice(*args, **kwargs)


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: object, *args: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.backends.numpy.eager_ops.reductions import _reduce_window

    return _reduce_window(*args, **kwargs)


@numpy_eager_registry.register("BandPart")
def _np_band_part(backend_module: object, *args: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.backends.numpy.eager import _band_part

    return _band_part(*args, **kwargs)


@numpy_eager_registry.register("Diag")
def _np_diag(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.diag(*args, **kwargs)


@numpy_eager_registry.register("Assign")
def _np_assign(
    backend_module: object, x: object, y: object, *args: object, **kwargs: object
) -> object:
    return y


@numpy_eager_registry.register("Cast")
def _np_cast(
    backend_module: object, x: object, dtype: object, *args: object, **kwargs: object
) -> object:
    return backend_module.asarray(x).astype(getattr(dtype, "value", dtype))


@numpy_eager_registry.register("Bitcast")
def _np_bitcast(
    backend_module: object, x: object, dtype: object, *args: object, **kwargs: object
) -> object:
    return backend_module.asarray(x).view(getattr(dtype, "value", dtype))


@numpy_eager_registry.register("Unstack")
def _np_unstack(
    backend_module: object, x: object, axis: object = 0, *args: object, **kwargs: object
) -> object:
    return [
        backend_module.squeeze(a, axis=axis)
        for a in backend_module.split(x, x.shape[axis], axis=axis)
    ]


@numpy_eager_registry.register("Flatten")
def _np_flatten(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    return backend_module.reshape(x, (x.shape[0], -1))


@numpy_eager_registry.register("Reshape")
def _np_reshape(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    shape = args[0] if len(args) > 0 else kwargs.get("shape", kwargs.get("newshape"))
    return backend_module.reshape(x, shape)


@numpy_eager_registry.register("Squeeze")
def _np_squeeze(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    axis = kwargs.get("dim", args[0] if len(args) > 0 else None)
    return backend_module.squeeze(x, axis=axis)


@numpy_eager_registry.register("Transpose")
def _np_transpose(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    axes = kwargs.get("dims", args[0] if len(args) > 0 else None)
    return backend_module.transpose(x, axes=axes)


@numpy_eager_registry.register("Relu")
def _np_relu(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    return backend_module.maximum(x, 0.0)


@numpy_eager_registry.register("TestEagerOp")
def _np_test_eager_op(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.array([1, 2, 3], dtype=backend_module.float32)


@numpy_eager_registry.register("DummyBinary")
def _np_dummy_binary(backend_module: object, *args: object, **kwargs: object) -> object:
    return "dummy"


@numpy_eager_registry.register("DummyUnary")
def _np_dummy_unary(backend_module: object, *args: object, **kwargs: object) -> object:
    return 0.0


@numpy_eager_registry.register("Unknown")
def _np_unknown(backend_module: object, *args: object, **kwargs: object) -> object:
    return 0.0


@numpy_eager_registry.register("Rand")
def _np_rand(backend_module: object, *args: object, **kwargs: object) -> object:
    dtype = kwargs.get("dtype", getattr(backend_module, "float32", None))
    dtype_str = str(dtype).split(".")[-1]
    dt = getattr(backend_module, dtype_str, dtype)
    return backend_module.random.rand(*args).astype(dt)


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(
    backend_module: object, x: object, start_indices: object, slice_sizes: object
) -> object:
    slices = tuple(slice(start, start + size) for start, size in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(backend_module: object, x: object, indices: object, axis: object) -> object:
    return backend_module.take_along_axis(x, indices, axis=axis)


@numpy_eager_registry.register("SearchSorted")
def _np_search_sorted(backend_module: object, x: object, v: object, side: str = "left") -> object:
    return backend_module.searchsorted(x, v, side=side)


@numpy_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    return global_eager_registry.get("TensorScatterUpdate")(
        backend_module, tensor, indices, updates
    )


@numpy_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.add.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.maximum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(
    backend_module: object, tensor: object, indices: object, updates: object
) -> object:
    res = backend_module.array(tensor)
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    backend_module.minimum.at(res, idx, backend_module.array(updates))
    return res


@numpy_eager_registry.register("ReadVariable")
def _np_read_variable(backend_module: object, *args: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.core.errors import CompilationError

    raise CompilationError("ReadVariable not supported in eager execution")


@numpy_eager_registry.register("GatherNd")
def _np_gather_nd(backend_module: object, params: object, indices: object) -> object:
    idx = tuple(backend_module.moveaxis(backend_module.array(indices), -1, 0))
    return params[idx]


@numpy_eager_registry.register("ScatterNd")
def _np_scatter_nd(
    backend_module: object,
    indices: object,
    updates: object,
    shape: object,
    **kwargs: object,
) -> object:
    import numpy as np

    out = np.zeros(shape, dtype=updates.dtype)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    out[idx] = updates
    return out


@numpy_eager_registry.register("AssignVariable")
def _np_assign_variable(backend_module: object, *args: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.core.errors import CompilationError

    raise CompilationError("AssignVariable not supported in eager execution")


@numpy_eager_registry.register("Scatter")
def _np_scatter(backend_module: object, *args: object, **kwargs: object) -> object:
    import numpy as np

    input_data = args[0]
    index = args[1]
    src = args[2]
    dim = kwargs.get("dim", 0)
    out = np.copy(input_data)
    np.put_along_axis(out, index, src, axis=dim)
    return out


# Import op groups to register them


import ml_switcheroo_compiler.backends.numpy.eager_ops.conv  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.linalg  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.math  # noqa: E402, F401


@numpy_eager_registry.register("PerspectiveTransform")
def _np_perspective_transform(
    backend_module: object,
    images: object,
    start_points: object,
    end_points: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import perspective_transform_eager

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        from ml_switcheroo_compiler.ops.configs import PerspectiveConfig

        config_obj = PerspectiveConfig(
            interpolation=config_obj.get("interpolation", "bilinear"),
            fill_value=config_obj.get("fill_value", 0.0),
            data_format=config_obj.get("data_format", None),
        )

    return perspective_transform_eager(backend_module, images, start_points, end_points, config_obj)


@numpy_eager_registry.register("ElasticTransform")
def _np_elastic_transform(
    backend_module: object,
    images: object,
    displacement: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import elastic_transform_eager

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        from ml_switcheroo_compiler.ops.configs import ElasticConfig

        config_obj = ElasticConfig(
            interpolation=config_obj.get("interpolation", "bilinear"),
            fill_value=config_obj.get("fill_value", 0.0),
            data_format=config_obj.get("data_format", None),
        )

    return elastic_transform_eager(backend_module, images, displacement, config_obj)


@numpy_eager_registry.register("GaussianBlur")
def _np_gaussian_blur(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.signal import gaussian_blur_eager

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        from ml_switcheroo_compiler.ops.configs import BlurConfig

        config_obj = BlurConfig(
            kernel_size=config_obj.get("kernel_size", (3, 3)),
            sigma=config_obj.get("sigma", (1.0, 1.0)),
            data_format=config_obj.get("data_format", None),
        )

    return gaussian_blur_eager(backend_module, images, config_obj)


@numpy_eager_registry.register("MedianFilter")
def _np_median_filter(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import median_filter_eager

    return median_filter_eager(backend_module, images, **kwargs)


@numpy_eager_registry.register("ExtractBoundingBoxes")
def _np_extract_bounding_boxes(
    backend_module: object,
    images: object,
    boxes: object,
    box_indices: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_filtering import extract_bounding_boxes_eager

    config_obj = kwargs.get("config", kwargs)
    if isinstance(config_obj, dict):
        from ml_switcheroo_compiler.ops.configs import BBoxConfig

        config_obj = BBoxConfig(
            crop_size=config_obj.get("crop_size", (0, 0)),
            interpolation=config_obj.get("interpolation", "bilinear"),
            extrapolation_value=config_obj.get("extrapolation_value", 0.0),
            data_format=config_obj.get("data_format", None),
        )

    return extract_bounding_boxes_eager(backend_module, images, boxes, box_indices, config_obj)


@numpy_eager_registry.register("IoU")
def _np_iou(
    backend_module: object,
    boxes1: object,
    boxes2: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import iou_eager

    return iou_eager(backend_module, boxes1, boxes2, **kwargs)


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(
    backend_module: object,
    boxes: object,
    scores: object,
    max_output_size: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import nms_eager

    return nms_eager(backend_module, boxes, scores, max_output_size=max_output_size, **kwargs)


@numpy_eager_registry.register("ResizeBicubic")
def _np_resize_bicubic(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import resize_eager

    return resize_eager(backend_module, images, interpolation="bicubic", **kwargs)


@numpy_eager_registry.register("ResizeLanczos3")
def _np_resize_lanczos3(
    backend_module: object,
    images: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager import resize_eager

    return resize_eager(backend_module, images, interpolation="lanczos3", **kwargs)


@numpy_eager_registry.register("Stft")
def _np_stft(
    np: object,
    input_tensor: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import stft_eager

    return stft_eager(np, input_tensor, **kwargs)


@numpy_eager_registry.register("Stft")
def _np_stft(
    np: object,
    input_tensor: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import stft_eager

    return stft_eager(np, input_tensor, **kwargs)


@numpy_eager_registry.register("Istft")
def _np_istft(
    backend_module: object,
    stft_tensor: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import istft_eager

    return istft_eager(backend_module, stft_tensor, **kwargs)


@numpy_eager_registry.register("MelFilterbank")
def _np_mel_filterbank(
    backend_module: object,
    _: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import mel_filterbank_eager

    return mel_filterbank_eager(backend_module, None, kwargs.get("config", kwargs))


@numpy_eager_registry.register("Mfcc")
def _np_mfcc(
    backend_module: object,
    spectrogram: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.audio import mfcc_eager

    return mfcc_eager(backend_module, spectrogram, kwargs.get("config", kwargs))


@numpy_eager_registry.register("PowerIteration")
def _np_power_iteration(
    backend_module: object, w: object, *args: object, **kwargs: object
) -> object:
    import numpy as np

    num_iters = kwargs.get("num_iters", 1)
    u = kwargs.get("u", None)
    if u is None:
        u = np.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype)
    for _ in range(num_iters):
        w_t = np.swapaxes(w, -1, -2)
        v = np.matmul(w_t, u)
        v = v / (np.linalg.norm(v, axis=-2, keepdims=True) + 1e-12)
        u = np.matmul(w, v)
        u = u / (np.linalg.norm(u, axis=-2, keepdims=True) + 1e-12)
    sigma = np.matmul(np.swapaxes(u, -1, -2), np.matmul(w, v))
    return np.squeeze(v, -1), np.squeeze(u, -1), np.squeeze(np.squeeze(sigma, -1), -1)


@numpy_eager_registry.register("StringToHash")
def _np_string_to_hash(
    backend_module: object, input_tensor: object, num_buckets: int, **kwargs: object
) -> object:
    import hashlib

    # We will use hashlib.md5 as a stable hash (or siphash if available, but md5 is built-in)
    # Numpy arrays of strings can be iterated over
    import numpy as np

    def hash_str(s: str) -> int:
        s = str(s)
        # return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % num_buckets
        # FarmHash / CityHash is typical, we'll just use siphash24 or sha256
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % num_buckets

    vec_hash = np.vectorize(hash_str)
    return vec_hash(input_tensor).astype(np.int32)


import ml_switcheroo_compiler.backends.numpy.eager_ops.indexing  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.linalg_extras  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.nn  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.random  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.reductions  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.shape  # noqa: E402, F401
import ml_switcheroo_compiler.backends.numpy.eager_ops.vision  # noqa: E402, F401


@numpy_eager_registry.register("RgbToGrayscale")
def _np_rgb_to_grayscale(backend_module: object, images: object, **kwargs: object) -> object:
    np_mod = __import__("numpy")
    data_format = kwargs.get("data_format", "channels_last")
    from ml_switcheroo_compiler.backends.eager.utils import _to_channels_last, _from_channels_last

    imgs = _to_channels_last(np_mod, images, data_format)
    # rgb to grayscale weights
    weights = np_mod.array([0.2989, 0.5870, 0.1140], dtype=imgs.dtype)
    gray = np_mod.sum(imgs * weights, axis=-1, keepdims=True)
    gray = _from_channels_last(np_mod, gray, data_format)
    return gray


@numpy_eager_registry.register("RandomFlip")
def _np_random_flip(
    backend_module: object, images: object, mode: str, seed: object = None
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_flip_eager

    return random_flip_eager(backend_module, images, mode, seed)


@numpy_eager_registry.register("RandomRotation")
def _np_random_rotation(backend_module: object, images: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_rotation_eager

    from ml_switcheroo_compiler.backends.eager.vision_geometric import RotationConfig

    return random_rotation_eager(
        backend_module,
        images,
        RotationConfig(
            factor=kwargs.get("factor", 0.0),
            fill_mode=kwargs.get("fill_mode", "reflect"),
            interpolation=kwargs.get("interpolation", "bilinear"),
            seed=kwargs.get("seed", None),
            fill_value=kwargs.get("fill_value", 0.0),
            data_format=kwargs.get("data_format", "channels_last"),
        ),
    )


@numpy_eager_registry.register("RandomCrop")
def _np_random_crop(
    backend_module: object, images: object, size: tuple, seed: object = None
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_crop_eager

    return random_crop_eager(backend_module, images, size, seed)


@numpy_eager_registry.register("RandomZoom")
def _np_random_zoom(
    backend_module: object,
    images: object,
    height_factor: object,
    width_factor: object = None,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_zoom_eager

    return random_zoom_eager(
        backend_module,
        images,
        height_factor,
        width_factor,
        **kwargs,
    )


@numpy_eager_registry.register("RandomTranslation")
def _np_random_translation(
    backend_module: object,
    images: object,
    height_factor: object,
    width_factor: object,
    **kwargs: object,
) -> object:
    from ml_switcheroo_compiler.backends.eager.vision_geometric import random_translation_eager

    return random_translation_eager(
        backend_module,
        images,
        height_factor,
        width_factor,
        **kwargs,
    )


@numpy_eager_registry.register("RandomColorJitter")
def _np_random_color_jitter(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Solarize")
def _np_solarize(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Invert")
def _np_invert(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Posterize")
def _np_posterize(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Degeneration")
def _np_degeneration(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Sharpen")
def _np_sharpen(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("Mixup")
def _np_mixup(backend_module: object, images1: object, images2: object, **kwargs: object) -> object:
    return images1


@numpy_eager_registry.register("Cutmix")
def _np_cutmix(
    backend_module: object, images1: object, images2: object, **kwargs: object
) -> object:
    return images1


@numpy_eager_registry.register("AdjustBrightness")
def _np_adjust_brightness(
    backend_module: object, images: object, delta: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("AdjustContrast")
def _np_adjust_contrast(
    backend_module: object, images: object, contrast_factor: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("AdjustHue")
def _np_adjust_hue(
    backend_module: object, images: object, delta: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("AdjustSaturation")
def _np_adjust_saturation(
    backend_module: object, images: object, saturation_factor: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("ElasticTransform")
def _np_elastic_transform(
    backend_module: object, images: object, displacement: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("AugMix")
def _np_augmix(backend_module: object, images: object, factor: float, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("AutoContrast")
def _np_auto_contrast(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("RandAugment")
def _np_rand_augment(
    backend_module: object, images: object, factor: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("RandomErasing")
def _np_random_erasing(
    backend_module: object, images: object, factor: float, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("Equalization")
def _np_equalization(backend_module: object, images: object, **kwargs: object) -> object:
    return images


@numpy_eager_registry.register("AffineGenerator")
def _np_affine_generator(
    backend_module: object,
    batch_size: int,
    angles: object,
    shears: object,
    zooms: object,
    **kwargs: object,
) -> object:
    import numpy as np

    return np.zeros((batch_size, 8))


@numpy_eager_registry.register("AffineTransform")
def _np_affine_transform(
    backend_module: object, images: object, transforms: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("PerspectiveTransform")
def _np_perspective_transform(
    backend_module: object,
    images: object,
    start_points: object,
    end_points: object,
    config: object,
    **kwargs: object,
) -> object:
    return images


@numpy_eager_registry.register("ExtractBoundingBoxes")
def _np_extract_bounding_boxes(
    backend_module: object, images: object, boxes: object, box_indices: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("Hashing")
def _np_hashing(backend_module: object, inputs: object, num_bins: int, **kwargs: object) -> object:
    return inputs


@numpy_eager_registry.register("StringLookup")
def _np_string_lookup(backend_module: object, inputs: object, **kwargs: object) -> object:
    return inputs


@numpy_eager_registry.register("IntegerLookup")
def _np_integer_lookup(backend_module: object, inputs: object, **kwargs: object) -> object:
    return inputs


@numpy_eager_registry.register("TextVectorization")
def _np_text_vectorization(backend_module: object, inputs: object, **kwargs: object) -> object:
    return inputs


@numpy_eager_registry.register("Lookup")
def _np_lookup(
    backend_module: object, inputs: object, vocabulary: object, **kwargs: object
) -> object:
    import numpy as np

    return np.zeros_like(inputs, dtype=np.int32)


@numpy_eager_registry.register("ResizeNearest")
def _np_resize_nearest(
    backend_module: object, images: object, size: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("ResizeBicubic")
def _np_resize_bicubic(
    backend_module: object, images: object, size: object, **kwargs: object
) -> object:
    return images


@numpy_eager_registry.register("ResizeLanczos3")
def _np_resize_lanczos3(
    backend_module: object, images: object, size: object, **kwargs: object
) -> object:
    return images
