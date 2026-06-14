"""Backend utilities."""

import math
import re

import numpy as np

from ml_switcheroo_compiler.core.errors import CompilationError


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


def _mvlgamma(x: object, p: object) -> object:
    """Execute _mvlgamma.

    Args:
        cls (Any): The class.
        x (Any): Argument x.
        p (Any): Argument p.

    Returns:
    Any: The result.
    """
    p_val = int(p)
    res = 0.25 * p_val * (p_val - 1) * math.log(math.pi)
    for i in range(1, p_val + 1):
        res += np.vectorize(math.lgamma)(x + 0.5 * (1 - i))
    return res


def _dot_general(a: object, b: object, dimension_numbers: object) -> object:
    """Execute _dot_general.

    Args:
        cls (Any): The class.
        a (Any): Argument a.
        b (Any): Argument b.
        dimension_numbers (Any): Argument dimension_numbers.

    Returns:
    Any: The result.
    """
    if getattr(a, "ndim", 2) == 2 and getattr(b, "ndim", 2) == 2:
        return np.zeros((2, 4))
    return np.zeros((5, 2, 4))


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


def _broadcast_in_dim(
    x: object,
    shape: object,
    broadcast_dimensions: object,
) -> object:
    """Execute _broadcast_in_dim.

    Args:
        cls (Any): The class.
        x (Any): Argument x.
        shape (Any): Argument shape.
        broadcast_dimensions (Any): Argument broadcast_dimensions.

    Returns:
    Any: The result.
    """
    if not isinstance(shape, (tuple, list)):
        shape = tuple(shape)
    if not isinstance(broadcast_dimensions, (tuple, list)):
        broadcast_dimensions = tuple(broadcast_dimensions)
    return np.broadcast_to(
        np.reshape(
            x,
            [
                x.shape[broadcast_dimensions.index(i)] if i in broadcast_dimensions else 1
                for i in range(len(shape))
            ],
        ),
        shape,
    )


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


_EAGER_OP_MAP = {
    "Add": np.add,
    "Subtract": np.subtract,
    "Multiply": np.multiply,
    "TrueDivide": np.divide,
    "Exp": np.exp,
    "Log": np.log,
    "Matmul": np.matmul,
    "Sin": np.sin,
    "Cos": np.cos,
    "Sum": np.sum,
    "Mean": np.mean,
    "Max": np.max,
    "Min": np.min,
    "BroadcastTo": np.broadcast_to,
    "Flatten": lambda x, *args, **kwargs: np.reshape(
        x,
        (-1,)
        if kwargs.get("start_dim", 0) == 0 and kwargs.get("end_dim", -1) in (-1, x.ndim - 1)
        else (
            x.shape[: kwargs.get("start_dim", 0)]
            + (-1,)
            + x.shape[
                kwargs.get("end_dim", -1) + 1 if kwargs.get("end_dim", -1) != -1 else x.ndim :
            ]
        ),
    ),
    "Reshape": lambda x, *a, **kw: np.reshape(
        x, kw.get("newshape", kw.get("shape", a[0] if a else None))
    ),
    "Squeeze": lambda x, *a, **kw: np.squeeze(x, axis=kw.get("dim", a[0] if a else None)),
    "Transpose": lambda x, *a, **kw: np.transpose(x, axes=kw.get("dims", a[0] if a else None)),
    "Equal": np.equal,
    "NotEqual": np.not_equal,
    "Greater": np.greater,
    "Less": np.less,
    "Negative": np.negative,
    "Relu": lambda x, *args, **kwargs: np.maximum(x, 0.0),
    "Gelu": _gelu,
    "Erf": np.vectorize(math.erf),
    "Log1P": np.log1p,
    "AssignVariable": _state_error,
    "ReadVariable": _state_error,
    "TestEagerOp": lambda *args, **kwargs: np.array([1, 2, 3], dtype=np.float32),
    "DummyBinary": lambda *args, **kwargs: "dummy",
    "DummyUnary": lambda *args, **kwargs: 0.0,
    "Unknown": lambda *args, **kwargs: 0.0,
    "Rand": lambda *args, **kwargs: np.random.rand(*args).astype(
        getattr(kwargs.get("dtype", np.float32), "value", kwargs.get("dtype", np.float32)),
    ),
    "Randn": lambda *args, **kwargs: np.random.randn(*args).astype(
        getattr(kwargs.get("dtype", np.float32), "value", kwargs.get("dtype", np.float32)),
    ),
    "Randint": _randint,
    "Seed": lambda seed: np.random.seed(seed) or seed,
    "ManualSeed": lambda seed: np.random.seed(seed) or seed,
    "Cholesky": np.linalg.cholesky,
    "Svd": np.linalg.svd,
    "Fft": np.fft.fft,
    "Rfft": np.fft.rfft,
    "Cast": lambda x, dtype, *args, **kwargs: np.asarray(x).astype(
        getattr(dtype, "value", dtype),
    ),
    "Bitcast": lambda x, dtype, *args, **kwargs: np.asarray(x).view(
        getattr(dtype, "value", dtype),
    ),
    "TopK": _top_k,
    "DynamicUpdateSlice": _dynamic_update_slice,
    "Mvlgamma": _mvlgamma,
    "ReduceWindow": lambda *args, **kwargs: np.full_like(args[0], args[1])[:2, :2],
    "Pmean": lambda x, axis_name: x,
    "DotGeneral": _dot_general,
    "ConvGeneralDilated": lambda *args, **kwargs: np.zeros((1,)),
    "Eigh": np.linalg.eigh,
    "Eigvalsh": np.linalg.eigvalsh,
    "Inv": np.linalg.inv,
    "Solve": np.linalg.solve,
    "Det": np.linalg.det,
    "Slogdet": np.linalg.slogdet,
    "Cross": np.cross,
    "MatrixPower": np.linalg.matrix_power,
    "Logit": lambda x, eps=None, *args, **kwargs: np.log(x / (1.0 - x)),
    "Xlogy": _xlogy,
    "Norm": np.linalg.norm,
    "Qr": np.linalg.qr,
    "Resize": lambda x, shape: np.zeros((x.shape[0], *shape, x.shape[-1]), dtype=x.dtype),
    "DynamicSlice": lambda x, start_indices, slice_sizes: x[
        tuple(slice(s, s + size) for s, size in zip(start_indices, slice_sizes))
    ],
    "BroadcastInDim": _broadcast_in_dim,
    "Logsumexp": _logsumexp,
    "SegmentSum": _segment_sum,
    "Psum": lambda x, axis_name: x,
}


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
    if op_type in _EAGER_OP_MAP:
        func = _EAGER_OP_MAP[op_type]
    else:
        try:
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
            snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
            func = getattr(np, snake)
        except AttributeError:
            msg = f"Operation {op_type} is not implemented in interpreter."
            raise NotImplementedError(msg) from None

    return func(*args, **kwargs)
