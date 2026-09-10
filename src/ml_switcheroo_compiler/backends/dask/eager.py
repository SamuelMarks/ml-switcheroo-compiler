"""Dedicated eager execution handlers and dispatch for the Dask backend."""

import sys
import types
from typing import Optional

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

try:
    import dask.array as da
except ImportError:
    da = None


def _verify_dask_task_graph(res: object) -> None:
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


def propagate_chunk_shapes(arr: object, chunks: Optional[object] = "auto") -> tuple[tuple[int, ...], ...]:
    """Calculate chunk structure and shape propagation for distributed arrays.

    Args:
        arr (object): Input Dask or sequence array.
        chunks (Optional[object]): Target chunking strategy ('auto' or tuple).

    Returns:
        tuple[tuple[int, ...], ...]: Standardized chunk tuple layout.
    """
    if hasattr(arr, "chunks") and arr.chunks is not None:
        return tuple(arr.chunks)
    if da is not None:
        try:
            dask_arr = da.from_array(arr, chunks=chunks)
            return tuple(dask_arr.chunks)
        except Exception:
            return ()
    return ()


def dask_conv2d(x: object, w: object, stride: object = 1, padding: object = "SAME", **kwargs: object) -> object:
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

    n, _, _, _ = x.shape
    out_c, _, kh, kw = w.shape
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    pad_h = kh // 2 if padding == "SAME" else 0
    pad_w = kw // 2 if padding == "SAME" else 0
    if pad_h > 0 or pad_w > 0:
        x_pad = da.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    else:
        x_pad = x
    out_h = (x_pad.shape[2] - kh) // stride_h + 1
    out_w = (x_pad.shape[3] - kw) // stride_w + 1
    chunks = (x.chunks[0] if hasattr(x, "chunks") and len(x.chunks) > 0 else (n,), (out_c,), (out_h,), (out_w,))
    out = da.zeros((n, out_c, out_h, out_w), dtype=x.dtype, chunks=chunks)
    for i in range(kh):
        for j in range(kw):
            patch = x_pad[:, :, i : i + out_h * stride_h : stride_h, j : j + out_w * stride_w : stride_w]
            out = out + da.tensordot(patch, w[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
    return out


def dask_maxpool2d(x: object, kernel_size: object = 2, stride: object = 2, padding: object = "VALID", **kwargs: object) -> object:
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
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
    sh, sw = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    out_h = (x.shape[2] - kh) // sh + 1
    out_w = (x.shape[3] - kw) // sw + 1
    out = x[:, :, 0 : out_h * sh : sh, 0 : out_w * sw : sw]
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            patch = x[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = da.maximum(out, patch)
    return out


def dask_avgpool2d(x: object, kernel_size: object = 2, stride: object = 2, padding: object = "VALID", **kwargs: object) -> object:
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
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
    sh, sw = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    out_h = (x.shape[2] - kh) // sh + 1
    out_w = (x.shape[3] - kw) // sw + 1
    out = x[:, :, 0 : out_h * sh : sh, 0 : out_w * sw : sw]
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            patch = x[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = out + patch
    return out / float(kh * kw)


def dask_resize_bilinear(x: object, size: object = None, **kwargs: object) -> object:
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
    new_h, new_w = (size[0], size[1])
    _, _, h, w = x.shape
    if (new_h, new_w) == (h, w):
        return x

    y_step: float = float(h - 1) / float(max(1, new_h - 1)) if new_h > 1 else 0.0
    x_step: float = float(w - 1) / float(max(1, new_w - 1)) if new_w > 1 else 0.0

    y_coords: list[float] = [i * y_step for i in range(new_h)]
    x_coords: list[float] = [j * x_step for j in range(new_w)]

    y0: list[int] = [int(v) for v in y_coords]
    y1: list[int] = [min(v + 1, h - 1) for v in y0]
    x0: list[int] = [int(v) for v in x_coords]
    x1: list[int] = [min(v + 1, w - 1) for v in x0]

    wy = da.from_array([v - float(int(v)) for v in y_coords], chunks=new_h)[None, None, :, None]
    wx = da.from_array([v - float(int(v)) for v in x_coords], chunks=new_w)[None, None, None, :]

    p00 = x[:, :, y0, :][:, :, :, x0]
    p01 = x[:, :, y0, :][:, :, :, x1]
    p10 = x[:, :, y1, :][:, :, :, x0]
    p11 = x[:, :, y1, :][:, :, :, x1]

    return p00 * (1.0 - wy) * (1.0 - wx) + p01 * (1.0 - wy) * wx + p10 * wy * (1.0 - wx) + p11 * wy * wx


def _attach_dask_vision_ops(backend_mod: Optional[object]) -> None:
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


def execute_op(  # noqa: C901, PLR0912
    cls: type,
    op_type: str,
    *args: object,
    **kwargs: object,
) -> object:
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
        backend_mod: Optional[object] = mod_entry
    else:
        backend_mod = da
    _attach_dask_vision_ops(backend_mod)

    schema = mapping_loader.load_backend_mappings("dask")
    if op_type in schema.operations:
        op_spec = schema.operations[op_type]
        is_custom = getattr(op_spec, "target_api", None) == "custom_op" and getattr(op_spec, "custom_code", None)

        if not is_custom and backend_mod is None:
            raise BackendNotSupportedError("Dask is not installed or available in this environment.")

        processed_args: list[object] = []
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
            pass

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(backend_mod, *args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.")
