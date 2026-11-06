"""Dedicated eager execution handlers and dispatch for the CuPy backend."""

from __future__ import annotations

import logging
import sys
import typing

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_eager_tensor, ensure_shape
from ml_switcheroo_compiler.backends.protocols import resolve_backend_member
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.shape import (
    DataHolderProtocol,
    DeviceHolderProtocol,
    extract_dtype,
    extract_shape,
)

try:
    import cupy as cp
except ImportError:
    cp = None


@typing.runtime_checkable
class _NdimProtocol(typing.Protocol):
    """Protocol for arrays exposing `.ndim`."""

    ndim: int


@typing.runtime_checkable
class _CupyVisionModuleProtocol(typing.Protocol):
    """Protocol for CuPy module containers supporting vision op attachment."""

    conv2d: EagerValue
    maxpool2d: EagerValue
    avgpool2d: EagerValue
    resize_bilinear: EagerValue


def _is_cupy_vision_mod(mod: EagerValue) -> typing.TypeGuard[_CupyVisionModuleProtocol]:
    """Type guard narrowing `mod` to `_CupyVisionModuleProtocol`.

    Args:
        mod (EagerValue): Candidate module or mock.

    Returns:
        bool: True if non-None.
    """
    return mod is not None


def _ndim_of(obj: EagerValue) -> int:
    """Return `.ndim` of `obj`, or `0` if not present.

    Args:
        obj (EagerValue): Input array or scalar.

    Returns:
        int: Number of dimensions.
    """
    return int(obj.ndim) if isinstance(obj, _NdimProtocol) else 0


def _verify_gpu_residency(res: EagerValue) -> None:
    """Verify that a CuPy tensor resides strictly in GPU device memory without host transfer.

    Args:
        res (EagerValue): Object or tensor resulting from CuPy operation.

    Raises:
        RuntimeError: If array lacks valid device memory residence.
    """
    if cp is None:
        return

    raw_res: EagerValue = res
    cp_ndarray = resolve_backend_member(cp, "ndarray", None)[1]
    is_arr = id(type(res)) == id(cp_ndarray) or isinstance(raw_res, DataHolderProtocol)
    if is_arr:
        data_buf: EagerValue = raw_res.data if isinstance(raw_res, DataHolderProtocol) else None
        if data_buf is not None and isinstance(data_buf, DeviceHolderProtocol):
            if data_buf.device is None:
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

    if _ndim_of(x) == 4 and _ndim_of(w) == 4:
        n, _, _, _ = extract_shape(x)
        out_c, _, kh, kw = extract_shape(w)
        stride_seq = ensure_shape(stride)
        stride_h, stride_w = (stride, stride) if isinstance(stride, int) else (int(stride_seq[0]), int(stride_seq[1]))
        pad_h = kh // 2 if padding == "SAME" else 0
        pad_w = kw // 2 if padding == "SAME" else 0
        if pad_h > 0 or pad_w > 0:
            x_pad = cp.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        else:
            x_pad = x
        out_h = (int(x_pad.shape[2]) - kh) // stride_h + 1
        out_w = (int(x_pad.shape[3]) - kw) // stride_w + 1
        out = cp.zeros((n, out_c, out_h, out_w), dtype=extract_dtype(x))
        for i in range(kh):
            for j in range(kw):
                patch = ensure_eager_tensor(x_pad)[:, :, i : i + out_h * stride_h : stride_h, j : j + out_w * stride_w : stride_w]
                out += cp.tensordot(patch, ensure_eager_tensor(w)[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
        return ensure_eager_tensor(out)

    if _ndim_of(x) == 2 and _ndim_of(w) == 2:
        kh, kw = extract_shape(w)
        pad_h = kh // 2 if padding == "SAME" else 0
        pad_w = kw // 2 if padding == "SAME" else 0
        x_pad = cp.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant") if (pad_h > 0 or pad_w > 0) else x
        x_pad_shape = extract_shape(x_pad)
        out_h = int(x_pad_shape[0]) - kh + 1
        out_w = int(x_pad_shape[1]) - kw + 1
        out = cp.zeros((out_h, out_w), dtype=extract_dtype(x))
        for i in range(kh):
            for j in range(kw):
                out += ensure_eager_tensor(x_pad)[i : i + out_h, j : j + out_w] * ensure_eager_tensor(w)[i, j]
        return ensure_eager_tensor(out)

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
    if cp is None or _ndim_of(x) != 4:
        return x

    k_seq = ensure_shape(kernel_size)
    s_seq = ensure_shape(stride)
    x_shape = extract_shape(x)
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    x_arr = ensure_eager_tensor(x)
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
    if cp is None or _ndim_of(x) != 4:
        return x

    k_seq = ensure_shape(kernel_size)
    s_seq = ensure_shape(stride)
    x_shape = extract_shape(x)
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    x_arr = ensure_eager_tensor(x)
    out = cp.zeros((x_shape[0], x_shape[1], out_h, out_w), dtype=extract_dtype(x))
    for i in range(kh):
        for j in range(kw):
            patch = x_arr[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out += patch
    return ensure_eager_tensor(out / float(kh * kw))


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
    if cp is None or _ndim_of(x) != 4 or size is None:
        return x

    size_seq = ensure_shape(size)
    new_h, new_w = (size_seq[0], size_seq[1])
    _, _, h, w = extract_shape(x)
    if (new_h, new_w) == (h, w):
        return ensure_eager_tensor(ensure_eager_tensor(x).copy())

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

    x_arr = ensure_eager_tensor(x)
    p00 = x_arr[:, :, y0[:, None], x0[None, :]]
    p01 = x_arr[:, :, y0[:, None], x1[None, :]]
    p10 = x_arr[:, :, y1[:, None], x0[None, :]]
    p11 = x_arr[:, :, y1[:, None], x1[None, :]]

    return ensure_eager_tensor(p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11)


def _attach_vision_ops(backend_mod: EagerValue | None) -> None:
    """Attach native vision and neural network kernels to CuPy backend container.

    Args:
        backend_mod (Optional[object]): The CuPy module or container.
    """
    if not _is_cupy_vision_mod(backend_mod):
        return
    if not resolve_backend_member(backend_mod, "conv2d")[0]:
        backend_mod.conv2d = ensure_eager_tensor(cupy_conv2d)
    if not resolve_backend_member(backend_mod, "maxpool2d")[0]:
        backend_mod.maxpool2d = ensure_eager_tensor(cupy_maxpool2d)
    if not resolve_backend_member(backend_mod, "avgpool2d")[0]:
        backend_mod.avgpool2d = ensure_eager_tensor(cupy_avgpool2d)
    if not resolve_backend_member(backend_mod, "resize_bilinear")[0]:
        backend_mod.resize_bilinear = ensure_eager_tensor(cupy_resize_bilinear)


def _prepare_cupy_args(args: tuple[EagerValue, ...], backend_mod: EagerValue | None) -> list[EagerValue]:
    """Convert sequence arguments into CuPy GPU arrays when appropriate.

    Args:
        args (tuple[object, ...]): Raw positional arguments.
        backend_mod (Optional[object]): CuPy module container.

    Returns:
        list[object]: Processed arguments list.
    """
    processed: list[EagerValue] = []
    has_asarray, asarray_fn = resolve_backend_member(backend_mod, "asarray") if backend_mod is not None else (False, None)
    for arg in args:
        if isinstance(arg, (list, tuple)) and has_asarray:
            try:
                processed.append(ensure_callable(asarray_fn)(arg))
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
        is_custom = op_spec.target_api == "custom_op" and bool(op_spec.custom_code)

        if not is_custom and backend_mod is None:
            mock_mod = sys.modules.get(__name__)
            target_api = op_spec.target_api or ""
            target_name = target_api.split(".")[-1]
            if mock_mod is not None and resolve_backend_member(mock_mod, target_name)[0]:
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
