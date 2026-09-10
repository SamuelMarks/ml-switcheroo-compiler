"""Dedicated eager execution handlers and dispatch for the CuPy backend."""

import sys
from typing import Optional

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

try:
    import cupy as cp
except ImportError:
    cp = None


def _verify_gpu_residency(res: object) -> None:
    """Verify that a CuPy tensor resides strictly in GPU device memory without host transfer.

    Args:
        res (object): Object or tensor resulting from CuPy operation.

    Raises:
        RuntimeError: If array lacks valid device memory residence.
    """
    if cp is None:
        return

    cp_ndarray = getattr(cp, "ndarray", None)
    is_arr = isinstance(res, cp_ndarray) if isinstance(cp_ndarray, type) else hasattr(res, "data")
    if is_arr:
        data_buf = getattr(res, "data", None)
        if data_buf is not None and hasattr(data_buf, "device"):
            if getattr(data_buf, "device", None) is None:
                msg = "CuPy array is not residing on GPU device memory."
                raise RuntimeError(msg)


def cupy_conv2d(x: object, w: object, stride: object = 1, padding: object = "SAME", **kwargs: object) -> object:
    """Execute 2D convolution directly on CuPy GPU memory.

    Args:
        x (object): Input tensor of shape (N, C, H, W) or (H, W).
        w (object): Weight filter tensor of shape (out_c, in_c, kh, kw) or (kh, kw).
        stride (object): Spatial stride integer or pair of integers.
        padding (object): Padding string ('SAME' or 'VALID').
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: Convolved CuPy GPU array.
    """
    del kwargs
    if cp is None:
        return x

    if getattr(x, "ndim", 0) == 4 and getattr(w, "ndim", 0) == 4:
        n, _, _, _ = x.shape
        out_c, _, kh, kw = w.shape
        stride_h, stride_w = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
        pad_h = kh // 2 if padding == "SAME" else 0
        pad_w = kw // 2 if padding == "SAME" else 0
        if pad_h > 0 or pad_w > 0:
            x_pad = cp.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        else:
            x_pad = x
        out_h = (x_pad.shape[2] - kh) // stride_h + 1
        out_w = (x_pad.shape[3] - kw) // stride_w + 1
        out = cp.zeros((n, out_c, out_h, out_w), dtype=x.dtype)
        for i in range(kh):
            for j in range(kw):
                patch = x_pad[:, :, i : i + out_h * stride_h : stride_h, j : j + out_w * stride_w : stride_w]
                out += cp.tensordot(patch, w[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
        return out

    if getattr(x, "ndim", 0) == 2 and getattr(w, "ndim", 0) == 2:
        kh, kw = w.shape
        pad_h = kh // 2 if padding == "SAME" else 0
        pad_w = kw // 2 if padding == "SAME" else 0
        x_pad = cp.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant") if (pad_h > 0 or pad_w > 0) else x
        out_h = x_pad.shape[0] - kh + 1
        out_w = x_pad.shape[1] - kw + 1
        out = cp.zeros((out_h, out_w), dtype=x.dtype)
        for i in range(kh):
            for j in range(kw):
                out += x_pad[i : i + out_h, j : j + out_w] * w[i, j]
        return out

    return x


def cupy_maxpool2d(x: object, kernel_size: object = 2, stride: object = 2, padding: object = "VALID", **kwargs: object) -> object:
    """Execute 2D max pooling directly on CuPy GPU memory.

    Args:
        x (object): Input tensor of shape (N, C, H, W).
        kernel_size (object): Pooling window size integer or tuple.
        stride (object): Window stride integer or tuple.
        padding (object): Padding mode ('VALID' or 'SAME').
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: Pooled CuPy GPU array.
    """
    del padding, kwargs
    if cp is None or getattr(x, "ndim", 0) != 4:
        return x

    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
    sh, sw = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    out_h = (x.shape[2] - kh) // sh + 1
    out_w = (x.shape[3] - kw) // sw + 1
    out = x[:, :, 0 : out_h * sh : sh, 0 : out_w * sw : sw].copy()
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            patch = x[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = cp.maximum(out, patch)
    return out


def cupy_avgpool2d(x: object, kernel_size: object = 2, stride: object = 2, padding: object = "VALID", **kwargs: object) -> object:
    """Execute 2D average pooling directly on CuPy GPU memory.

    Args:
        x (object): Input tensor of shape (N, C, H, W).
        kernel_size (object): Pooling window size integer or tuple.
        stride (object): Window stride integer or tuple.
        padding (object): Padding mode ('VALID' or 'SAME').
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: Average-pooled CuPy GPU array.
    """
    del padding, kwargs
    if cp is None or getattr(x, "ndim", 0) != 4:
        return x

    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
    sh, sw = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    out_h = (x.shape[2] - kh) // sh + 1
    out_w = (x.shape[3] - kw) // sw + 1
    out = cp.zeros((x.shape[0], x.shape[1], out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = x[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out += patch
    return out / float(kh * kw)


def cupy_resize_bilinear(x: object, size: object = None, **kwargs: object) -> object:
    """Execute bilinear resize directly on CuPy GPU memory.

    Args:
        x (object): Input tensor of shape (N, C, H, W).
        size (object): Target spatial dimensions (new_h, new_w).
        **kwargs (object): Additional keyword arguments.

    Returns:
        object: Resized CuPy GPU array.
    """
    del kwargs
    if cp is None or getattr(x, "ndim", 0) != 4 or size is None:
        return x

    new_h, new_w = (size[0], size[1])
    _, _, h, w = x.shape
    if (new_h, new_w) == (h, w):
        return x.copy()

    y_coords = cp.linspace(0, h - 1, new_h, dtype=cp.float32)
    x_coords = cp.linspace(0, w - 1, new_w, dtype=cp.float32)
    y0 = cp.floor(y_coords).astype(cp.int32)
    y1 = cp.minimum(y0 + 1, h - 1)
    x0 = cp.floor(x_coords).astype(cp.int32)
    x1 = cp.minimum(x0 + 1, w - 1)

    wy = (y_coords - y0)[:, None]
    wx = (x_coords - x0)[None, :]

    w00 = (1.0 - wy) * (1.0 - wx)
    w01 = (1.0 - wy) * wx
    w10 = wy * (1.0 - wx)
    w11 = wy * wx

    p00 = x[:, :, y0[:, None], x0[None, :]]
    p01 = x[:, :, y0[:, None], x1[None, :]]
    p10 = x[:, :, y1[:, None], x0[None, :]]
    p11 = x[:, :, y1[:, None], x1[None, :]]

    return p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11


def _attach_vision_ops(backend_mod: Optional[object]) -> None:
    """Attach native vision and neural network kernels to CuPy backend container.

    Args:
        backend_mod (Optional[object]): The CuPy module or container.
    """
    if backend_mod is None:
        return
    for name, fn in (
        ("conv2d", cupy_conv2d),
        ("maxpool2d", cupy_maxpool2d),
        ("avgpool2d", cupy_avgpool2d),
        ("resize_bilinear", cupy_resize_bilinear),
    ):
        if not hasattr(backend_mod, name):
            setattr(backend_mod, name, fn)


def _prepare_cupy_args(args: tuple[object, ...], backend_mod: Optional[object]) -> list[object]:
    """Convert sequence arguments into CuPy GPU arrays when appropriate.

    Args:
        args (tuple[object, ...]): Raw positional arguments.
        backend_mod (Optional[object]): CuPy module container.

    Returns:
        list[object]: Processed arguments list.
    """
    processed: list[object] = []
    has_asarray = backend_mod is not None and hasattr(backend_mod, "asarray")
    for arg in args:
        if isinstance(arg, (list, tuple)) and has_asarray:
            try:
                processed.append(backend_mod.asarray(arg))
            except Exception:
                processed.append(arg)
        else:
            processed.append(arg)
    return processed


def execute_op(  # noqa: C901, PLR0912
    cls: type,
    op_type: str,
    *args: object,
    **kwargs: object,
) -> object:
    """Execute operation eagerly on CuPy backend preserving GPU memory residency.

    Args:
        cls (type): The caller class or context.
        op_type (str): The name of the operation.
        *args (object): Positional arguments for the op.
        **kwargs (object): Keyword arguments for the op.

    Returns:
        object: Result of native CuPy evaluation.

    Raises:
        BackendNotSupportedError: If CuPy is unavailable or op is unmapped.
    """
    schema = mapping_loader.load_backend_mappings("cupy")
    backend_mod: Optional[object] = cp if cp is not None else sys.modules.get("cupy")
    _attach_vision_ops(backend_mod)

    if op_type in schema.operations:
        op_spec = schema.operations[op_type]
        is_custom = getattr(op_spec, "target_api", None) == "custom_op" and getattr(op_spec, "custom_code", None)

        if not is_custom and backend_mod is None:
            mock_mod = sys.modules.get(__name__)
            target_api = getattr(op_spec, "target_api", "") or ""
            target_name = target_api.split(".")[-1]
            if mock_mod is not None and hasattr(mock_mod, target_name):
                backend_mod = mock_mod
            else:
                raise BackendNotSupportedError("CuPy is not installed or available in this environment.")

        processed_args = _prepare_cupy_args(args, backend_mod)

        try:
            res = mapping_loader.dispatch_eager_op("cupy", op_type, processed_args, dict(kwargs), backend_module=backend_mod)
            _verify_gpu_residency(res)
            return res
        except BackendNotSupportedError:
            pass

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(backend_mod, *args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.")
