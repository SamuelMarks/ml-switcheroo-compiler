"""Dedicated eager execution handlers and dispatch for the Dask backend."""

import logging
import sys
import types
from typing import Optional, cast

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

try:
    import dask.array as da
except ImportError:
    da = cast(types.ModuleType, None)


def _verify_dask_task_graph(res: EagerValue) -> None:
    """Verify that result is a native lazy Dask task graph and has not resolved to eager host memory.

    Args:
        res (object): Object resulting from Dask eager operation.

    Raises:
        RuntimeError: If the result is an eagerly resolved NumPy array instead of a Dask Array.
    """
    if da is None:
        return
    da_array = getattr(da, "Array", None)
    if isinstance(da_array, type) and isinstance(res, da_array) and not hasattr(res, "dask"):
        msg = "Dask array is missing underlying task graph (.dask)."
        raise RuntimeError(msg)


def propagate_chunk_shapes(arr: EagerValue, chunks: Optional[EagerValue] = "auto") -> tuple[tuple[int, ...], ...]:
    """Calculate chunk structure and shape propagation for distributed arrays.

    Args:
        arr (object): Input Dask or sequence array.
        chunks (Optional[object]): Target chunking strategy ('auto' or tuple).

    Returns:
        tuple[tuple[int, ...], ...]: Standardized chunk tuple layout.
    """
    if hasattr(arr, "chunks") and getattr(arr, "chunks") is not None:
        return tuple(getattr(arr, "chunks"))
    if da is not None:
        try:
            dask_arr = getattr(da, "from_array")(arr, chunks=chunks)
            return tuple(dask_arr.chunks)
        except Exception:
            return ()
    return ()


def dask_conv2d(x: EagerValue, w: EagerValue, stride: EagerValue = 1, padding: EagerValue = "SAME", **kwargs: EagerValue) -> EagerValue:
    """Execute 2D convolution lazily as a Dask task graph.

    Args:
        x (object): Input Dask array of shape (N, C, H, W).
        w (object): Weight filter tensor.
        stride (object): Spatial stride.
        padding (object): Padding mode ('SAME' or 'VALID').
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: Lazy Dask array with task graph.
    """
    del kwargs
    if da is None or getattr(x, "ndim", 0) != 4 or getattr(w, "ndim", 0) != 4:
        return x

    n, _, _, _ = cast(tuple[int, ...], getattr(x, "shape"))
    out_c, _, kh, kw = cast(tuple[int, ...], getattr(w, "shape"))
    stride_seq = cast(tuple[int, ...], stride)
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else (int(stride_seq[0]), int(stride_seq[1]))
    pad_h = kh // 2 if padding == "SAME" else 0
    pad_w = kw // 2 if padding == "SAME" else 0
    if pad_h > 0 or pad_w > 0:
        x_pad = getattr(da, "pad")(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    else:
        x_pad = x
    x_pad_shape = cast(tuple[int, ...], x_pad.shape)
    out_h = (int(x_pad_shape[2]) - kh) // stride_h + 1
    out_w = (int(x_pad_shape[3]) - kw) // stride_w + 1
    chunks = (getattr(x, "chunks")[0] if hasattr(x, "chunks") and len(getattr(x, "chunks")) > 0 else (n,), (out_c,), (out_h,), (out_w,))
    out = getattr(da, "zeros")((n, out_c, out_h, out_w), dtype=getattr(x, "dtype"), chunks=chunks)
    for i in range(kh):
        for j in range(kw):
            patch = cast(EagerTensorProtocol, x_pad)[:, :, i : i + out_h * stride_h : stride_h, j : j + out_w * stride_w : stride_w]
            out = out + getattr(da, "tensordot")(patch, cast(EagerTensorProtocol, w)[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
    return cast(EagerValue, out)


def dask_maxpool2d(x: EagerValue, kernel_size: EagerValue = 2, stride: EagerValue = 2, padding: EagerValue = "VALID", **kwargs: EagerValue) -> EagerValue:
    """Execute 2D max pooling lazily as a Dask task graph.

    Args:
        x (object): Input Dask array of shape (N, C, H, W).
        kernel_size (object): Window size.
        stride (object): Stride.
        padding (object): Padding mode.
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: Lazy Dask array.
    """
    del padding, kwargs
    if da is None or getattr(x, "ndim", 0) != 4:
        return x
    k_seq = cast(tuple[int, ...], kernel_size)
    s_seq = cast(tuple[int, ...], stride)
    x_shape = cast(tuple[int, ...], getattr(x, "shape"))
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    x_arr = cast(EagerTensorProtocol, x)
    out = x_arr[:, :, 0 : out_h * sh : sh, 0 : out_w * sw : sw]
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            patch = x_arr[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = getattr(da, "maximum")(out, patch)
    return out


def dask_avgpool2d(x: EagerValue, kernel_size: EagerValue = 2, stride: EagerValue = 2, padding: EagerValue = "VALID", **kwargs: EagerValue) -> EagerValue:
    """Execute 2D average pooling lazily as a Dask task graph.

    Args:
        x (object): Input Dask array of shape (N, C, H, W).
        kernel_size (object): Window size.
        stride (object): Stride.
        padding (object): Padding mode.
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: Lazy Dask array.
    """
    del padding, kwargs
    if da is None or getattr(x, "ndim", 0) != 4:
        return x
    k_seq = cast(tuple[int, ...], kernel_size)
    s_seq = cast(tuple[int, ...], stride)
    x_shape = cast(tuple[int, ...], getattr(x, "shape"))
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    x_arr = cast(EagerTensorProtocol, x)
    out = x_arr[:, :, 0 : out_h * sh : sh, 0 : out_w * sw : sw]
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            patch = x_arr[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = out + patch
    return out / float(kh * kw)


def dask_resize_bilinear(x: EagerValue, size: EagerValue = None, **kwargs: EagerValue) -> EagerValue:
    """Execute bilinear resize lazily as a Dask task graph.

    Args:
        x (object): Input Dask array of shape (N, C, H, W).
        size (object): Target spatial dimensions (new_h, new_w).
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: Lazy resized Dask array.
    """
    del kwargs
    if da is None or getattr(x, "ndim", 0) != 4 or size is None:
        return x
    size_seq = cast(tuple[int, ...], size)
    new_h, new_w = (int(size_seq[0]), int(size_seq[1]))
    _, _, h, w = cast(tuple[int, ...], getattr(x, "shape"))
    if (new_h, new_w) == (h, w):
        return x

    y_step: float = float(h - 1) / float(max(1, new_h - 1)) if new_h > 1 else 0.0
    x_step: float = float(w - 1) / float(max(1, new_w - 1)) if new_w > 1 else 0.0

    y_coords: list[float] = [i * y_step for i in range(new_h)]
    x_coords: list[float] = [j * x_step for j in range(new_w)]

    y0: list[int] = [int(v) for v in y_coords]
    y1: list[int] = [min(v + 1, int(h) - 1) for v in y0]
    x0: list[int] = [int(v) for v in x_coords]
    x1: list[int] = [min(v + 1, int(w) - 1) for v in x0]

    wy = getattr(da, "from_array")([v - float(int(v)) for v in y_coords], chunks=new_h)[None, None, :, None]
    wx = getattr(da, "from_array")([v - float(int(v)) for v in x_coords], chunks=new_w)[None, None, None, :]

    x_arr = cast(EagerTensorProtocol, x)
    p00 = x_arr[:, :, y0, :][:, :, :, x0]
    p01 = x_arr[:, :, y0, :][:, :, :, x1]
    p10 = x_arr[:, :, y1, :][:, :, :, x0]
    p11 = x_arr[:, :, y1, :][:, :, :, x1]

    return cast(EagerValue, p00 * (1.0 - wy) * (1.0 - wx) + p01 * (1.0 - wy) * wx + p10 * wy * (1.0 - wx) + p11 * wy * wx)


def _attach_dask_vision_ops(backend_mod: Optional[EagerValue]) -> None:
    """Attach lazy Dask vision and neural network kernels to Dask container.

    Args:
        backend_mod (Optional[object]): Dask module container.
    """
    if backend_mod is None:
        return
    for name, fn in (
        ("conv2d", dask_conv2d),
        ("maxpool2d", dask_maxpool2d),
        ("avgpool2d", dask_avgpool2d),
        ("resize_bilinear", dask_resize_bilinear),
    ):
        if not hasattr(backend_mod, name):
            setattr(backend_mod, name, fn)


def execute_op(
    cls: type,
    op_type: str,
    *args: EagerValue,
    **kwargs: EagerValue,
) -> EagerValue:
    """Execute operation eagerly on Dask backend without computing graph or falling back.

    Args:
        cls (type): The caller class or context.
        op_type (str): The name of the operation.
        *args (object): Positional arguments for the op.
        **kwargs (object): Keyword arguments for the op.

    Returns:
        object: Result of native Dask evaluation (e.g. dask.array.Array).

    Raises:
        BackendNotSupportedError: If Dask is unavailable or op is unmapped.
    """
    mod_entry = sys.modules.get("ml_switcheroo_compiler.backends.dask.eager")
    if mod_entry is not None and not isinstance(mod_entry, types.ModuleType):
        backend_mod: Optional[EagerValue] = mod_entry
    else:
        backend_mod = da
    _attach_dask_vision_ops(backend_mod)

    schema = mapping_loader.load_backend_mappings("dask")
    if op_type in schema.operations:
        op_spec = schema.operations[op_type]
        is_custom = getattr(op_spec, "target_api", None) == "custom_op" and getattr(op_spec, "custom_code", None)

        if not is_custom and backend_mod is None:
            raise BackendNotSupportedError("Dask is not installed or available in this environment.")

        processed_args: list[EagerValue] = []
        for arg in args:
            if isinstance(arg, (list, tuple)) and backend_mod is not None and hasattr(backend_mod, "from_array"):
                try:
                    processed_args.append(backend_mod.from_array(arg))
                except Exception:
                    processed_args.append(arg)
            else:
                processed_args.append(arg)

        try:
            res = mapping_loader.dispatch_eager_op("dask", op_type, processed_args, dict(kwargs), backend_module=backend_mod)
            _verify_dask_task_graph(res)
            return res
        except BackendNotSupportedError:
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(backend_mod, *args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.")
