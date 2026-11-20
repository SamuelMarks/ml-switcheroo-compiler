"""Dedicated eager execution handlers and dispatch for the CuPy backend."""

from __future__ import annotations

import logging
import sys
import types
import typing

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendArrayProtocol,
    EagerTensorProtocol,
    coerce_array,
    coerce_shape,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
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


class _CupyVisionModuleProxy(types.SimpleNamespace):
    """Explicit delegation wrapper exposing CuPy vision kernels alongside ``inner`` module attributes."""

    def __init__(self, inner: BackendArg) -> None:
        """Initialize proxy around ``inner`` CuPy module or container.

        Args:
            inner (BackendArg): Underlying CuPy module or container.
        """
        attrs: dict[str, BackendArg] = {
            name: adapt_backend_module(inner).resolve_symbol(name)[1]
            for name in dir(inner)
            if not name.startswith("__") and adapt_backend_module(inner).resolve_symbol(name)[0]
        }
        attrs.setdefault("conv2d", cupy_conv2d)
        attrs.setdefault("maxpool2d", cupy_maxpool2d)
        attrs.setdefault("avgpool2d", cupy_avgpool2d)
        attrs.setdefault("resize_bilinear", cupy_resize_bilinear)
        super().__init__(**attrs)


def _ndim_of(obj: BackendArg) -> int:
    """Return `.ndim` of `obj`, or `0` if not present.

    Args:
        obj (BackendArg): Input array or scalar.

    Returns:
        int: Number of dimensions.
    """
    return int(obj.ndim) if isinstance(obj, _NdimProtocol) else 0


def _verify_gpu_residency(res: BackendArg) -> None:
    """Verify that a CuPy tensor resides strictly in GPU device memory without host transfer.

    Args:
        res (BackendArg): Object or tensor resulting from CuPy operation.

    Raises:
        RuntimeError: If array lacks valid device memory residence.
    """
    if cp is None:
        return

    raw_res: BackendArg = res
    cp_ndarray = adapt_backend_module(cp).resolve_symbol("ndarray", None)[1]
    is_arr = id(type(res)) == id(cp_ndarray) or isinstance(raw_res, DataHolderProtocol)
    if is_arr:
        data_buf: BackendArg = raw_res.data if isinstance(raw_res, DataHolderProtocol) else None
        if data_buf is not None and isinstance(data_buf, DeviceHolderProtocol):
            if data_buf.device is None:
                msg = "CuPy array is not residing on GPU device memory."
                raise RuntimeError(msg)


def cupy_conv2d(
    x: BackendArg, w: BackendArg, stride: BackendArg = 1, padding: BackendArg = "SAME", **kwargs: BackendArg
) -> BackendArg:
    """Execute 2D convolution directly on CuPy GPU memory.

    Args:
        x (BackendArg): Input tensor of shape (N, C, H, W) or (H, W).
        w (BackendArg): Weight filter tensor of shape (out_c, in_c, kh, kw) or (kh, kw).
        stride (BackendArg): Spatial stride integer or pair of integers.
        padding (BackendArg): Padding string ('SAME' or 'VALID').
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Convolved CuPy GPU array.
    """
    del kwargs
    if cp is None:
        return x

    if _ndim_of(x) == 4 and _ndim_of(w) == 4:
        n, _, _, _ = extract_shape(x)
        out_c, _, kh, kw = extract_shape(w)
        stride_seq = coerce_shape(stride)
        stride_h, stride_w = (stride, stride) if isinstance(stride, int) else (int(stride_seq[0]), int(stride_seq[1]))
        pad_h = kh // 2 if padding == "SAME" else 0
        pad_w = kw // 2 if padding == "SAME" else 0
        if pad_h > 0 or pad_w > 0:
            x_pad: BackendArrayProtocol = require_callable(cp.pad)(
                x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
            )
        else:
            x_pad = coerce_array(cp, x)
        w_arr: BackendArrayProtocol = coerce_array(cp, w)
        out_h = (int(x_pad.shape[2]) - kh) // stride_h + 1
        out_w = (int(x_pad.shape[3]) - kw) // stride_w + 1
        out: BackendArrayProtocol = require_callable(cp.zeros)((n, out_c, out_h, out_w), dtype=extract_dtype(x))
        for i in range(kh):
            for j in range(kw):
                patch = x_pad[:, :, i : i + out_h * stride_h : stride_h, j : j + out_w * stride_w : stride_w]
                td: EagerTensorProtocol = require_callable(cp.tensordot)(patch, w_arr[:, :, i, j], axes=([1], [1]))
                out = out + adapt_backend_module(td).require_op("transpose")(0, 3, 1, 2)
        return out

    if _ndim_of(x) == 2 and _ndim_of(w) == 2:
        kh, kw = extract_shape(w)
        pad_h = kh // 2 if padding == "SAME" else 0
        pad_w = kw // 2 if padding == "SAME" else 0
        x_pad_2d: BackendArrayProtocol = (
            require_callable(cp.pad)(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
            if (pad_h > 0 or pad_w > 0)
            else coerce_array(cp, x)
        )
        w_2d: BackendArrayProtocol = coerce_array(cp, w)
        x_pad_shape = extract_shape(x_pad_2d)
        out_h = int(x_pad_shape[0]) - kh + 1
        out_w = int(x_pad_shape[1]) - kw + 1
        out_2d: BackendArrayProtocol = require_callable(cp.zeros)((out_h, out_w), dtype=extract_dtype(x))
        for i in range(kh):
            for j in range(kw):
                out_2d = out_2d + x_pad_2d[i : i + out_h, j : j + out_w] * w_2d[i, j]
        return out_2d

    return x


def cupy_maxpool2d(
    x: BackendArg,
    kernel_size: BackendArg = 2,
    stride: BackendArg = 2,
    padding: BackendArg = "VALID",
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute 2D max pooling directly on CuPy GPU memory.

    Args:
        x (BackendArg): Input tensor of shape (N, C, H, W).
        kernel_size (BackendArg): Pooling window size integer or tuple.
        stride (BackendArg): Window stride integer or tuple.
        padding (BackendArg): Padding mode ('VALID' or 'SAME').
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Pooled CuPy GPU array.
    """
    del padding, kwargs
    if cp is None or _ndim_of(x) != 4:
        return x

    k_seq = coerce_shape(kernel_size)
    s_seq = coerce_shape(stride)
    x_shape = extract_shape(x)
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    x_arr: BackendArrayProtocol = coerce_array(cp, x)
    out: BackendArrayProtocol = adapt_backend_module(x_arr[:, :, 0 : out_h * sh : sh, 0 : out_w * sw : sw]).require_op(
        "copy"
    )()
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            patch = x_arr[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = require_callable(cp.maximum)(out, patch)
    return out


def cupy_avgpool2d(
    x: BackendArg,
    kernel_size: BackendArg = 2,
    stride: BackendArg = 2,
    padding: BackendArg = "VALID",
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute 2D average pooling directly on CuPy GPU memory.

    Args:
        x (BackendArg): Input tensor of shape (N, C, H, W).
        kernel_size (BackendArg): Pooling window size integer or tuple.
        stride (BackendArg): Window stride integer or tuple.
        padding (BackendArg): Padding mode ('VALID' or 'SAME').
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Average-pooled CuPy GPU array.
    """
    del padding, kwargs
    if cp is None or _ndim_of(x) != 4:
        return x

    k_seq = coerce_shape(kernel_size)
    s_seq = coerce_shape(stride)
    x_shape = extract_shape(x)
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    x_arr: BackendArrayProtocol = coerce_array(cp, x)
    out: BackendArrayProtocol = require_callable(cp.zeros)(
        (x_shape[0], x_shape[1], out_h, out_w), dtype=extract_dtype(x)
    )
    for i in range(kh):
        for j in range(kw):
            patch = x_arr[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = out + patch
    return out / float(kh * kw)


def cupy_resize_bilinear(x: BackendArg, size: BackendArg = None, **kwargs: BackendArg) -> BackendArg:
    """Execute bilinear resize directly on CuPy GPU memory.

    Args:
        x (BackendArg): Input tensor of shape (N, C, H, W).
        size (BackendArg): Target spatial dimensions (new_h, new_w).
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Resized CuPy GPU array.
    """
    del kwargs
    if cp is None or _ndim_of(x) != 4 or size is None:
        return x

    size_seq = coerce_shape(size)
    new_h, new_w = (int(size_seq[0]), int(size_seq[1]))
    _, _, h, w = extract_shape(x)
    if (new_h, new_w) == (h, w):
        return adapt_backend_module(x).require_op("copy")()

    y_coords: EagerTensorProtocol = require_callable(cp.linspace)(0, int(h) - 1, new_h, dtype=cp.float32)
    x_coords: EagerTensorProtocol = require_callable(cp.linspace)(0, int(w) - 1, new_w, dtype=cp.float32)
    y0: EagerTensorProtocol = adapt_backend_module(require_callable(cp.floor)(y_coords)).require_op("astype")(cp.int32)
    y1: EagerTensorProtocol = require_callable(cp.minimum)(y0 + 1, int(h) - 1)
    x0: EagerTensorProtocol = adapt_backend_module(require_callable(cp.floor)(x_coords)).require_op("astype")(cp.int32)
    x1: EagerTensorProtocol = require_callable(cp.minimum)(x0 + 1, int(w) - 1)

    wy = (y_coords - y0)[:, None]
    wx = (x_coords - x0)[None, :]

    w00 = (1.0 - wy) * (1.0 - wx)
    w01 = (1.0 - wy) * wx
    w10 = wy * (1.0 - wx)
    w11 = wy * wx

    x_arr: BackendArrayProtocol = coerce_array(cp, x)
    p00 = x_arr[:, :, y0[:, None], x0[None, :]]
    p01 = x_arr[:, :, y0[:, None], x1[None, :]]
    p10 = x_arr[:, :, y1[:, None], x0[None, :]]
    p11 = x_arr[:, :, y1[:, None], x1[None, :]]

    return p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11


def _attach_vision_ops(backend_mod: BackendArg | None) -> BackendArg | None:
    """Wrap CuPy backend container with vision and neural network kernels.

    Args:
        backend_mod (BackendArg | None): The CuPy module or container.

    Returns:
        BackendArg | None: Proxy exposing vision ops, or ``None`` if ``backend_mod`` is ``None``.
    """
    if backend_mod is None:
        return None
    if adapt_backend_module(backend_mod).resolve_symbol("_mock_name")[0]:
        return backend_mod
    return _CupyVisionModuleProxy(backend_mod)


def _prepare_cupy_args(args: tuple[BackendArg, ...], backend_mod: BackendArg | None) -> list[BackendArg]:
    """Convert sequence arguments into CuPy GPU arrays when appropriate.

    Args:
        args (tuple[BackendArg, ...]): Raw positional arguments.
        backend_mod (BackendArg | None): CuPy module container.

    Returns:
        list[object]: Processed arguments list.
    """
    processed: list[BackendArg] = []
    has_asarray, asarray_fn = (
        adapt_backend_module(backend_mod).resolve_symbol("asarray") if backend_mod is not None else (False, None)
    )
    for arg in args:
        if isinstance(arg, (list, tuple)) and has_asarray:
            try:
                require_callable(processed.append)(require_callable(asarray_fn)(arg))
            except (
                AttributeError,
                ImportError,
                LookupError,
                OSError,
                ArithmeticError,
                RuntimeError,
                SyntaxError,
                TypeError,
                ValueError,
            ):
                processed.append(arg)
        else:
            processed.append(arg)
    return processed


def execute_op(
    cls: type | None,
    op_type: str,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute operation eagerly on CuPy backend preserving GPU memory residency.

    Args:
        cls (type | None): The caller class or context.
        op_type (str): The name of the operation.
        *args (BackendArg): Positional arguments for the op.
        **kwargs (BackendArg): Keyword arguments for the op.

    Returns:
        BackendArg: Result of native CuPy evaluation.

    Raises:
        BackendNotSupportedError: If CuPy is unavailable or op is unmapped.
    """
    schema = mapping_loader.load_backend_mappings("cupy")
    raw_mod: BackendArg | None = cp if cp is not None else sys.modules.get("cupy")
    backend_mod: BackendArg | None = _attach_vision_ops(raw_mod)

    if op_type in schema.operations:
        op_spec = schema.operations[op_type]
        is_custom = op_spec.target_api == "custom_op" and bool(op_spec.custom_code)

        if not is_custom and backend_mod is None:
            mock_mod = sys.modules.get(__name__)
            target_api = op_spec.target_api or ""
            target_name = target_api.split(".")[-1]
            if mock_mod is not None and adapt_backend_module(mock_mod).resolve_symbol(target_name)[0]:
                backend_mod = mock_mod
            else:
                raise BackendNotSupportedError("CuPy is not installed or available in this environment.")

        processed_args = _prepare_cupy_args(args, backend_mod)

        try:
            res = mapping_loader.dispatch_eager_op(
                "cupy", op_type, processed_args, dict(kwargs), backend_module=backend_mod
            )
            _verify_gpu_residency(res)
            return res
        except BackendNotSupportedError:
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(backend_mod, *args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.")
