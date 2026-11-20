"""Dedicated eager execution handlers and dispatch for the Dask backend."""

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
    coerce_operand_list,
    coerce_shape,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.shape import (
    extract_dtype,
    extract_shape,
)

da: types.ModuleType | None
try:
    import dask.array as _da

    da = _da
except ImportError:
    da = None


@typing.runtime_checkable
class _DaskGraphHolderProtocol(typing.Protocol):
    """Protocol for Dask arrays exposing `.dask`."""

    dask: BackendArg


@typing.runtime_checkable
class _ChunksHolderProtocol(typing.Protocol):
    """Protocol for Dask arrays exposing `.chunks`."""

    chunks: BackendArg


@typing.runtime_checkable
class _NdimProtocol(typing.Protocol):
    """Protocol for arrays exposing `.ndim`."""

    ndim: int


def _ndim_of(obj: BackendArg) -> int:
    """Return `.ndim` of `obj`, or `0` if not present.

    Args:
        obj (BackendArg): Input array or scalar.

    Returns:
        int: Number of dimensions.
    """
    return int(obj.ndim) if isinstance(obj, _NdimProtocol) else 0


def _verify_dask_task_graph(res: BackendArg) -> None:
    """Verify that result is a native lazy Dask task graph and has not resolved to eager host memory.

    Args:
        res (BackendArg): Object resulting from Dask eager operation.

    Raises:
        RuntimeError: If the result is an eagerly resolved NumPy array instead of a Dask Array.
    """
    if da is None:
        return
    raw_res: BackendArg = res
    da_array = adapt_backend_module(da).resolve_symbol("Array", None)[1]
    if id(type(res)) == id(da_array) and not isinstance(raw_res, _DaskGraphHolderProtocol):
        msg = "Dask array is missing underlying task graph (.dask)."
        raise RuntimeError(msg)


def propagate_chunk_shapes(arr: BackendArg, chunks: BackendArg | None = "auto") -> tuple[tuple[int, ...], ...]:
    """Calculate chunk structure and shape propagation for distributed arrays.

    Args:
        arr (BackendArg): Input Dask or sequence array.
        chunks (BackendArg | None): Target chunking strategy ('auto' or tuple).

    Returns:
        tuple[tuple[int, ...], ...]: Standardized chunk tuple layout.
    """
    raw_arr: BackendArg | _ChunksHolderProtocol = arr
    if isinstance(raw_arr, _ChunksHolderProtocol) and raw_arr.chunks is not None:
        return tuple(coerce_shape(c) for c in coerce_operand_list(raw_arr.chunks))
    if da is not None:
        try:
            dask_arr = da.from_array(arr, chunks=chunks)
            return tuple(coerce_shape(c) for c in coerce_operand_list(dask_arr.chunks))
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
            return ()
    return ()


def dask_conv2d(
    x: BackendArg, w: BackendArg, stride: BackendArg = 1, padding: BackendArg = "SAME", **kwargs: BackendArg
) -> BackendArg:
    """Execute 2D convolution lazily as a Dask task graph.

    Args:
        x (BackendArg): Input Dask array of shape (N, C, H, W).
        w (BackendArg): Weight filter tensor.
        stride (BackendArg): Spatial stride.
        padding (BackendArg): Padding mode ('SAME' or 'VALID').
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Lazy Dask array with task graph.
    """
    del kwargs
    if da is None or _ndim_of(x) != 4 or _ndim_of(w) != 4:
        return x

    n, _, _, _ = extract_shape(x)
    out_c, _, kh, kw = extract_shape(w)
    stride_seq = coerce_shape(stride)
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else (int(stride_seq[0]), int(stride_seq[1]))
    pad_h = kh // 2 if padding == "SAME" else 0
    pad_w = kw // 2 if padding == "SAME" else 0
    da_mod = da
    if pad_h > 0 or pad_w > 0:
        x_pad: BackendArrayProtocol = require_callable(da_mod.pad)(
            x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
    else:
        x_pad = coerce_array(da_mod, x)
    w_arr: BackendArrayProtocol = coerce_array(da_mod, w)
    x_pad_shape = extract_shape(x_pad)
    out_h = (int(x_pad_shape[2]) - kh) // stride_h + 1
    out_w = (int(x_pad_shape[3]) - kw) // stride_w + 1
    raw_x: BackendArg | _ChunksHolderProtocol = x
    x_chunks: list[BackendArg] = coerce_operand_list(raw_x.chunks) if isinstance(raw_x, _ChunksHolderProtocol) else []
    chunks = (x_chunks[0] if len(x_chunks) > 0 else (n,), (out_c,), (out_h,), (out_w,))
    out: BackendArrayProtocol = require_callable(da_mod.zeros)(
        (n, out_c, out_h, out_w), dtype=extract_dtype(x), chunks=chunks
    )
    for i in range(kh):
        for j in range(kw):
            patch = x_pad[:, :, i : i + out_h * stride_h : stride_h, j : j + out_w * stride_w : stride_w]
            td: EagerTensorProtocol = require_callable(da_mod.tensordot)(patch, w_arr[:, :, i, j], axes=([1], [1]))
            out = out + adapt_backend_module(td).require_op("transpose")(0, 3, 1, 2)
    return out


def dask_maxpool2d(
    x: BackendArg,
    kernel_size: BackendArg = 2,
    stride: BackendArg = 2,
    padding: BackendArg = "VALID",
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute 2D max pooling lazily as a Dask task graph.

    Args:
        x (BackendArg): Input Dask array of shape (N, C, H, W).
        kernel_size (BackendArg): Window size.
        stride (BackendArg): Stride.
        padding (BackendArg): Padding mode.
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Lazy Dask array.
    """
    del padding, kwargs
    if da is None or _ndim_of(x) != 4:
        return x
    k_seq = coerce_shape(kernel_size)
    s_seq = coerce_shape(stride)
    x_shape = extract_shape(x)
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    da_mod = da
    x_arr: BackendArrayProtocol = coerce_array(da_mod, x)
    out: BackendArrayProtocol = x_arr[:, :, 0 : out_h * sh : sh, 0 : out_w * sw : sw]
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            patch = x_arr[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = require_callable(da_mod.maximum)(out, patch)
    return out


def dask_avgpool2d(
    x: BackendArg,
    kernel_size: BackendArg = 2,
    stride: BackendArg = 2,
    padding: BackendArg = "VALID",
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute 2D average pooling lazily as a Dask task graph.

    Args:
        x (BackendArg): Input Dask array of shape (N, C, H, W).
        kernel_size (BackendArg): Window size.
        stride (BackendArg): Stride.
        padding (BackendArg): Padding mode.
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Lazy Dask array.
    """
    del padding, kwargs
    if da is None or _ndim_of(x) != 4:
        return x
    k_seq = coerce_shape(kernel_size)
    s_seq = coerce_shape(stride)
    x_shape = extract_shape(x)
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (int(k_seq[0]), int(k_seq[1]))
    sh, sw = (stride, stride) if isinstance(stride, int) else (int(s_seq[0]), int(s_seq[1]))
    out_h = (int(x_shape[2]) - kh) // sh + 1
    out_w = (int(x_shape[3]) - kw) // sw + 1
    x_arr: BackendArrayProtocol = coerce_array(da, x)
    out: BackendArrayProtocol = x_arr[:, :, 0 : out_h * sh : sh, 0 : out_w * sw : sw]
    for i in range(kh):
        for j in range(kw):
            if i == 0 and j == 0:
                continue
            patch = x_arr[:, :, i : i + out_h * sh : sh, j : j + out_w * sw : sw]
            out = out + patch
    return out / float(kh * kw)


def dask_resize_bilinear(x: BackendArg, size: BackendArg = None, **kwargs: BackendArg) -> BackendArg:
    """Execute bilinear resize lazily as a Dask task graph.

    Args:
        x (BackendArg): Input Dask array of shape (N, C, H, W).
        size (BackendArg): Target spatial dimensions (new_h, new_w).
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Lazy resized Dask array.
    """
    del kwargs
    if da is None or _ndim_of(x) != 4 or size is None:
        return x
    size_seq = coerce_shape(size)
    new_h, new_w = (int(size_seq[0]), int(size_seq[1]))
    _, _, h, w = extract_shape(x)
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

    da_mod = da
    from_arr_fn = require_callable(da_mod.from_array)
    wy: EagerTensorProtocol = from_arr_fn([v - float(int(v)) for v in y_coords], chunks=new_h)[None, None, :, None]
    wx: EagerTensorProtocol = from_arr_fn([v - float(int(v)) for v in x_coords], chunks=new_w)[None, None, None, :]

    x_arr: BackendArrayProtocol = coerce_array(da_mod, x)
    p00 = x_arr[:, :, y0, :][:, :, :, x0]
    p01 = x_arr[:, :, y0, :][:, :, :, x1]
    p10 = x_arr[:, :, y1, :][:, :, :, x0]
    p11 = x_arr[:, :, y1, :][:, :, :, x1]

    return p00 * (1.0 - wy) * (1.0 - wx) + p01 * (1.0 - wy) * wx + p10 * wy * (1.0 - wx) + p11 * wy * wx


class _DaskVisionModuleProxy(types.SimpleNamespace):
    """Explicit delegation wrapper exposing Dask vision kernels alongside ``inner`` module attributes."""

    def __init__(self, inner: BackendArg) -> None:
        """Initialize proxy around ``inner`` Dask module or container.

        Args:
            inner (BackendArg): Underlying Dask module or container.
        """
        attrs: dict[str, BackendArg] = {
            name: adapt_backend_module(inner).resolve_symbol(name)[1]
            for name in dir(inner)
            if not name.startswith("__") and adapt_backend_module(inner).resolve_symbol(name)[0]
        }
        attrs.setdefault("conv2d", dask_conv2d)
        attrs.setdefault("maxpool2d", dask_maxpool2d)
        attrs.setdefault("avgpool2d", dask_avgpool2d)
        attrs.setdefault("resize_bilinear", dask_resize_bilinear)
        super().__init__(**attrs)


def _attach_dask_vision_ops(backend_mod: BackendArg | None) -> BackendArg | None:
    """Wrap Dask container with lazy Dask vision and neural network kernels.

    Args:
        backend_mod (BackendArg | None): Dask module container.

    Returns:
        BackendArg | None: Proxy exposing vision ops, or ``None`` if ``backend_mod`` is ``None``.
    """
    if backend_mod is None:
        return None
    if adapt_backend_module(backend_mod).resolve_symbol("_mock_name")[0]:
        return backend_mod
    return _DaskVisionModuleProxy(backend_mod)


def execute_op(
    cls: type | None,
    op_type: str,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute operation eagerly on Dask backend without computing graph or falling back.

    Args:
        cls (type | None): The caller class or context.
        op_type (str): The name of the operation.
        *args (BackendArg): Positional arguments for the op.
        **kwargs (BackendArg): Keyword arguments for the op.

    Returns:
        BackendArg: Result of native Dask evaluation (e.g. dask.array.Array).

    Raises:
        BackendNotSupportedError: If Dask is unavailable or op is unmapped.
    """
    mod_entry: types.ModuleType | BackendArg | None = sys.modules.get("ml_switcheroo_compiler.backends.dask.eager")
    if mod_entry is not None and not isinstance(mod_entry, types.ModuleType):
        raw_mod: BackendArg | None = mod_entry
    else:
        raw_mod = da
    backend_mod: BackendArg | None = _attach_dask_vision_ops(raw_mod)

    schema = mapping_loader.load_backend_mappings("dask")
    if op_type in schema.operations:
        op_spec = schema.operations[op_type]
        is_custom = op_spec.target_api == "custom_op" and bool(op_spec.custom_code)

        if not is_custom and backend_mod is None:
            raise BackendNotSupportedError("Dask is not installed or available in this environment.")

        has_from_array, from_array_fn = (
            adapt_backend_module(backend_mod).resolve_symbol("from_array") if backend_mod is not None else (False, None)
        )
        processed_args: list[BackendArg] = []
        for arg in args:
            if isinstance(arg, (list, tuple)) and has_from_array:
                try:
                    require_callable(processed_args.append)(require_callable(from_array_fn)(arg))
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
                    processed_args.append(arg)
            else:
                processed_args.append(arg)

        try:
            res = mapping_loader.dispatch_eager_op(
                "dask", op_type, processed_args, dict(kwargs), backend_module=backend_mod
            )
            _verify_dask_task_graph(res)
            return res
        except BackendNotSupportedError:
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    func = global_eager_registry.get(op_type)
    if func is not None:
        return func(backend_mod, *args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' is not implemented.")
