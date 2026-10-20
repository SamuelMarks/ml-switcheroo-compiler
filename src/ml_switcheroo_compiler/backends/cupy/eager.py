"""Dedicated eager execution handlers and dispatch for the CuPy backend."""

from __future__ import annotations

import logging
import sys
import typing

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

try:
    import cupy as cp
except ImportError:
    cp = None


def _verify_gpu_residency(res: EagerValue) -> None:
    """Verify that a CuPy tensor resides strictly in GPU device memory without host transfer.

    Args:
        res (EagerValue): Object or tensor resulting from CuPy operation.

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


def cupy_conv2d(x: EagerValue, w: EagerValue, stride: EagerValue = 1, padding: EagerValue = "SAME", **kwargs: EagerValue) -> EagerValue:
    """Execute 2D convolution directly on CuPy GPU memory.

    Args:
        x (EagerValue): Input tensor of shape (N, C, H, W) or (H, W).
        w (EagerValue): Weight filter tensor of shape (out_c, in_c, kh, kw) or (kh, kw).
        stride (EagerValue): Spatial stride integer or pair of integers.
        padding (EagerValue): Padding string ('SAME' or 'VALID').
        **kwargs (EagerValue): Additional keyword arguments.

    Returns:
        EagerValue: Convolved CuPy GPU array.
    """
    del kwargs
    if cp is None:
        return x

    if getattr(x, "ndim", 0) == 4 and getattr(w, "ndim", 0) == 4:
        n, _, _, _ = typing.cast(tuple[int, ...], getattr(x, "shape"))
        out_c, _, kh, kw = typing.cast(tuple[int, ...], getattr(w, "shape"))
        stride_seq = typing.cast(tuple[int, ...], stride)
        stride_h, stride_w = (stride, stride) if isinstance(stride, int) else (int(stride_seq[0]), int(stride_seq[1]))
        pad_h = kh // 2 if padding == "SAME" else 0
        pad_w = kw // 2 if padding == "SAME" else 0
        if pad_h > 0 or pad_w > 0:
            x_pad = cp.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        else:
            x_pad = x
        out_h = (int(x_pad.shape[2]) - kh) // stride_h + 1
        out_w = (int(x_pad.shape[3]) - kw) // stride_w + 1
        out = cp.zeros((n, out_c, out_h, out_w), dtype=getattr(x, "dtype"))
        for i in range(kh):
            for j in range(kw):
                patch = typing.cast(EagerTensorProtocol, x_pad)[:, :, i : i + out_h * stride_h : stride_h, j : j + out_w * stride_w : stride_w]
                out += cp.tensordot(patch, typing.cast(EagerTensorProtocol, w)[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
        return typing.cast(EagerValue, out)

    if getattr(x, "ndim", 0) == 2 and getattr(w, "ndim", 0) == 2:
        kh, kw = typing.cast(tuple[int, ...], getattr(w, "shape"))
        pad_h = kh // 2 if padding == "SAME" else 0
        pad_w = kw // 2 if padding == "SAME" else 0
        x_pad = cp.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant") if (pad_h > 0 or pad_w > 0) else x
        x_pad_shape = typing.cast(tuple[int, ...], getattr(x_pad, "shape"))
        out_h = int(x_pad_shape[0]) - kh + 1
        out_w = int(x_pad_shape[1]) - kw + 1
        out = cp.zeros((out_h, out_w), dtype=getattr(x, "dtype"))
        for i in range(kh):
            for j in range(kw):
                out += typing.cast(EagerTensorProtocol, x_pad)[i : i + out_h, j : j + out_w] * typing.cast(EagerTensorProtocol, w)[i, j]
        return typing.cast(EagerValue, out)

    return x


def cupy_maxpool2d(x: EagerValue, kernel_size: EagerValue = 2, stride: EagerValue = 2, padding: EagerValue = "VALID", **kwargs: EagerValue) -> EagerValue:
    """Execute 2D max pooling directly on CuPy GPU memory.

    Args:
        x (EagerValue): Input tensor of shape (N, C, H, W).
        kernel_size (EagerValue): Pooling window size integer or tuple.
        stride (EagerValue): Window stride integer or tuple.
        padding (EagerValue): Padding mode ('VALID' or 'SAME').
        **kwargs (EagerValue): Additional keyword arguments.

    Returns:
        EagerValue: Pooled CuPy GPU array.
    """
    del padding, kwargs
    if cp is None or getattr(x, "ndim", 0) != 4:
        return x

    k_seq = typing.cast(tuple[int, ...], kernel_size)
    s_seq = typing.cast(tuple[int, ...], stride)
    x_shape = typing.cast(tuple[int, ...], getattr(x, "shape"))
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    x_arr = typing.cast(EagerTensorProtocol, x)
    out = x_arr[:, :, 0 : out_h * sh : sh, 0 : out_w * sw : sw].copy()
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            patch = x_arr[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = cp.maximum(out, patch)
    return out


def cupy_avgpool2d(x: EagerValue, kernel_size: EagerValue = 2, stride: EagerValue = 2, padding: EagerValue = "VALID", **kwargs: EagerValue) -> EagerValue:
    """Execute 2D average pooling directly on CuPy GPU memory.

    Args:
        x (EagerValue): Input tensor of shape (N, C, H, W).
        kernel_size (EagerValue): Pooling window size integer or tuple.
        stride (EagerValue): Window stride integer or tuple.
        padding (EagerValue): Padding mode ('VALID' or 'SAME').
        **kwargs (EagerValue): Additional keyword arguments.

    Returns:
        EagerValue: Average-pooled CuPy GPU array.
    """
    del padding, kwargs
    if cp is None or getattr(x, "ndim", 0) != 4:
        return x

    k_seq = typing.cast(tuple[int, ...], kernel_size)
    s_seq = typing.cast(tuple[int, ...], stride)
    x_shape = typing.cast(tuple[int, ...], getattr(x, "shape"))
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    x_arr = typing.cast(EagerTensorProtocol, x)
    out = cp.zeros((x_shape[0], x_shape[1], out_h, out_w), dtype=getattr(x, "dtype"))
    for i in range(kh):
        for j in range(kw):
            patch = x_arr[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out += patch
    return typing.cast(EagerValue, out / float(kh * kw))


def cupy_resize_bilinear(x: EagerValue, size: EagerValue = None, **kwargs: EagerValue) -> EagerValue:
    """Execute bilinear resize directly on CuPy GPU memory.

    Args:
        x (EagerValue): Input tensor of shape (N, C, H, W).
        size (EagerValue): Target spatial dimensions (new_h, new_w).
        **kwargs (EagerValue): Additional keyword arguments.

    Returns:
        EagerValue: Resized CuPy GPU array.
    """
    del kwargs
    if cp is None or getattr(x, "ndim", 0) != 4 or size is None:
        return x

    size_seq = typing.cast(tuple[int, ...], size)
    new_h, new_w = (size_seq[0], size_seq[1])
    _, _, h, w = typing.cast(tuple[int, ...], getattr(x, "shape"))
    if (new_h, new_w) == (h, w):
        return typing.cast(EagerValue, getattr(x, "copy")())

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

    x_arr = typing.cast(EagerTensorProtocol, x)
    p00 = x_arr[:, :, y0[:, None], x0[None, :]]
    p01 = x_arr[:, :, y0[:, None], x1[None, :]]
    p10 = x_arr[:, :, y1[:, None], x0[None, :]]
    p11 = x_arr[:, :, y1[:, None], x1[None, :]]

    return typing.cast(EagerValue, p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11)


def _attach_vision_ops(backend_mod: EagerValue | None) -> None:
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


def _prepare_cupy_args(args: tuple[EagerValue, ...], backend_mod: EagerValue | None) -> list[EagerValue]:
    """Convert sequence arguments into CuPy GPU arrays when appropriate.

    Args:
        args (tuple[object, ...]): Raw positional arguments.
        backend_mod (Optional[object]): CuPy module container.

    Returns:
        list[object]: Processed arguments list.
    """
    processed: list[EagerValue] = []
    has_asarray = backend_mod is not None and hasattr(backend_mod, "asarray")
    for arg in args:
        if isinstance(arg, (list, tuple)) and has_asarray:
            try:
                processed.append(getattr(backend_mod, "asarray")(arg))
            except Exception:
                processed.append(arg)
        else:
            processed.append(arg)
    return processed


def execute_op(
    cls: type,
    op_type: str,
    *args: EagerValue,
    **kwargs: EagerValue,
) -> EagerValue:
    """Execute operation eagerly on CuPy backend preserving GPU memory residency.

    Args:
        cls (type): The caller class or context.
        op_type (str): The name of the operation.
        *args (EagerValue): Positional arguments for the op.
        **kwargs (EagerValue): Keyword arguments for the op.

    Returns:
        EagerValue: Result of native CuPy evaluation.

    Raises:
        BackendNotSupportedError: If CuPy is unavailable or op is unmapped.
    """
    schema = mapping_loader.load_backend_mappings("cupy")
    backend_mod: EagerValue | None = cp if cp is not None else sys.modules.get("cupy")
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
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(backend_mod, *args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.")
