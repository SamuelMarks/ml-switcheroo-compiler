"""Generic backend-parametric eager reference operations for `reference_math.math_scatter`."""

from __future__ import annotations

import types
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
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


@global_eager_registry.register("TensorScatterUpdate")
def _np_tensor_scatter_update(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
) -> EagerTensorProtocol:
    """Evaluate tensor scatter update operation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        tensor (BackendArg): Target tensor.
        indices (BackendArg): Scatter indices.
        updates (BackendArg): Values to scatter.

    Returns:
        BackendArg: Updated tensor.
    """
    res = adapt_backend_module(backend_module).require_symbol("copy")(
        adapt_backend_module(backend_module).require_symbol("array")(tensor)
    )
    idx = tuple(
        adapt_backend_module(backend_module).require_symbol("moveaxis")(
            adapt_backend_module(backend_module).require_symbol("array")(indices), -1, 0
        )
    )
    res[idx] = adapt_backend_module(backend_module).require_symbol("array")(updates)
    return _ensure_arr(res)


@global_eager_registry.register("TensorScatterAdd")
def _np_tensor_scatter_add(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
) -> EagerTensorProtocol:
    """Evaluate tensor scatter add operation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        tensor (BackendArg): Target tensor.
        indices (BackendArg): Scatter indices.
        updates (BackendArg): Values to add.

    Returns:
        BackendArg: Tensor with accumulated updates.
    """
    res = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx = tuple(
        adapt_backend_module(backend_module).require_symbol("moveaxis")(
            adapt_backend_module(backend_module).require_symbol("array")(indices), -1, 0
        )
    )
    adapt_backend_module(backend_module).require_symbol("add").at(
        res, idx, adapt_backend_module(backend_module).require_symbol("array")(updates)
    )
    return _ensure_arr(res)


@global_eager_registry.register("TensorScatterMax")
def _np_tensor_scatter_max(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
) -> EagerTensorProtocol:
    """Evaluate tensor scatter maximum operation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        tensor (BackendArg): Target tensor.
        indices (BackendArg): Scatter indices.
        updates (BackendArg): Values to take maximum with.

    Returns:
        BackendArg: Tensor with maximum applied.
    """
    res = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx = tuple(
        adapt_backend_module(backend_module).require_symbol("moveaxis")(
            adapt_backend_module(backend_module).require_symbol("array")(indices), -1, 0
        )
    )
    adapt_backend_module(backend_module).require_symbol("maximum").at(
        res, idx, adapt_backend_module(backend_module).require_symbol("array")(updates)
    )
    return _ensure_arr(res)


@global_eager_registry.register("TensorScatterMin")
def _np_tensor_scatter_min(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    indices: BackendArg,
    updates: BackendArg,
) -> EagerTensorProtocol:
    """Evaluate tensor scatter minimum operation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        tensor (BackendArg): Target tensor.
        indices (BackendArg): Scatter indices.
        updates (BackendArg): Values to take minimum with.

    Returns:
        BackendArg: Tensor with minimum applied.
    """
    res = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx = tuple(
        adapt_backend_module(backend_module).require_symbol("moveaxis")(
            adapt_backend_module(backend_module).require_symbol("array")(indices), -1, 0
        )
    )
    adapt_backend_module(backend_module).require_symbol("minimum").at(
        res, idx, adapt_backend_module(backend_module).require_symbol("array")(updates)
    )
    return _ensure_arr(res)


@global_eager_registry.register("ScatterNd")
def _np_scatter_nd(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    indices: BackendArg,
    updates: BackendArg,
    shape: tuple[int, ...] | Sequence[int],
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Evaluate scatter_nd operation creating zero tensor with scattered updates.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        indices (BackendArg): Index coordinates.
        updates (BackendArg): Source updates tensor.
        shape (tuple[int, ...] | Sequence[int]): Output shape tuple.
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        BackendArg: Scattered tensor of specified shape.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    upd_arr = arr_mod.asarray(updates)
    out = arr_mod.zeros(tuple(shape), dtype=upd_arr.dtype)
    idx = tuple(arr_mod.moveaxis(arr_mod.array(indices), -1, 0))
    out[idx] = upd_arr
    return _ensure_arr(out)


@global_eager_registry.register("Scatter")
def _np_scatter(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg | int,
    **kwargs: int | None,
) -> EagerTensorProtocol:
    """Evaluate scatter operation along a dimension.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg | int):
            Positional arguments (input_data, index, src).
        **kwargs (Union[int, None]): Keyword arguments such as dim.

    Returns:
        BackendArg: Scattered array along specified axis.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    input_data = args[0]
    index = args[1]
    src = args[2]
    dim_val = kwargs.get("dim", 0)
    dim = int(dim_val) if dim_val is not None else 0
    out = arr_mod.copy(input_data)
    adapt_backend_module(arr_mod).require_symbol("put_along_axis")(out, index, src, axis=dim)
    return out


def _band_part(
    input_tensor: BackendArg,
    num_lower: int,
    num_upper: int,
) -> EagerTensorProtocol:
    """Evaluate band_part matrix operation.

    Args:
        input_tensor (BackendArg): Input tensor of matrices.
        num_lower (int): Number of subdiagonals to keep.
        num_upper (int): Number of superdiagonals to keep.

    Returns:
        BackendArg: Band-part extracted matrix tensor.
    """
    arr_mod = _resolve_eager_mod(None)
    input_arr = arr_mod.asarray(input_tensor)
    res = arr_mod.copy(input_arr)
    return res


@global_eager_registry.register("GatherNd")
def _np_gather_nd(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    params: BackendArg,
    indices: BackendArg,
) -> EagerTensorProtocol:
    """Execute the eager np gather nd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        params (BackendArg): Source parameters tensor.
        indices (BackendArg): Indices slice coordinates.

    Returns:
        BackendArg: Gathered tensor elements.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    idx = tuple(
        adapt_backend_module(backend_module).require_symbol("moveaxis")(
            adapt_backend_module(backend_module).require_symbol("array")(indices), -1, 0
        )
    )
    return _ensure_arr(coerce_array(backend_module, params)[idx])


@global_eager_registry.register("TakeAlongAxis")
def _np_take_along_axis(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    indices: BackendArg,
    axis: int,
) -> EagerTensorProtocol:
    """Execute the eager np take along axis operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Source values array.
        indices (BackendArg): 1D or ND indices array.
        axis (int): Axis along which to take.

    Returns:
        BackendArg: Selected slice values.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    x_arr = coerce_array(backend_module, x)
    idx_arr = coerce_array(backend_module, indices)
    if x_arr.ndim != idx_arr.ndim:
        return _ensure_arr(adapt_backend_module(backend_module).require_symbol("take")(x_arr, idx_arr, axis=axis))
    return _ensure_arr(
        adapt_backend_module(backend_module).require_symbol("take_along_axis")(x_arr, idx_arr, axis=axis)
    )


@global_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    start_indices: Sequence[int],
    slice_sizes: Sequence[int],
) -> EagerTensorProtocol:
    """Evaluate dynamic slicing from variable starts with fixed sizes.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Source tensor.
        start_indices (Sequence[int]): Starting coordinates per axis.
        slice_sizes (Sequence[int]): Size of slice per axis.

    Returns:
        BackendArg: Dynamically sliced subarray.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    slices = tuple(slice(start, start + size) for (start, size) in zip(start_indices, slice_sizes))
    return coerce_array(backend_module, x)[slices]


@global_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg | Sequence[int],
    **kwargs: float | int | None,
) -> EagerTensorProtocol:
    """Evaluate dynamic update slice operation.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg | Sequence[int]):
            Positional arguments (operand, update, start_indices).
        **kwargs (Union[float, int, None]): Additional keyword arguments.

    Returns:
        BackendArg: Updated array with slice replaced.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_shape

    arr_mod = _resolve_eager_mod(backend_module)
    x_arr = arr_mod.copy(args[0])
    upd = args[1]
    starts = coerce_operand_list(args[2])
    slcs = tuple(
        slice(
            int(
                coerce_int(
                    adapt_backend_module(s).require_symbol("item")()
                    if adapt_backend_module(s).has_symbol("item")
                    else s
                )
            ),
            int(
                coerce_int(
                    adapt_backend_module(s).require_symbol("item")()
                    if adapt_backend_module(s).has_symbol("item")
                    else s
                )
            )
            + int(sz),
        )
        for s, sz in zip(starts, coerce_shape(adapt_backend_module(upd).require_symbol("shape")))
    )
    x_arr[slcs] = upd
    return x_arr
