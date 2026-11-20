"""Operations and utilities for `ml_switcheroo_compiler.backends.mlx.eager`."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import types
import typing

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendArrayProtocol,
    EagerTensorProtocol,
    coerce_array,
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_numeric,
    coerce_operand_list,
    coerce_opt_int,
    coerce_shape,
    coerce_str,
    global_eager_registry,
    mlx_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

mx: types.ModuleType | BackendModuleProtocol | None
try:
    mx = importlib.import_module("mlx.core")
except ImportError:
    mx = None


def _get_mlx_func(op_type: str) -> BackendArg:
    """Retrieve the corresponding MLX function for a given operation type.

    Args:
        op_type (str): The name of the operation.

    Returns: mx.array: The MLX function if found, otherwise None.
    """
    import re

    s1: str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_type)
    snake: str = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    if snake == "mul":
        snake = "multiply"
    elif snake == "sub":
        snake = "subtract"
    elif snake == "div":
        snake = "divide"
    for mod in [
        mx,
        adapt_backend_module(mx).get_symbol("linalg", None),
        adapt_backend_module(mx).get_symbol("fft", None),
    ]:
        if mod is not None and adapt_backend_module(mod).has_symbol(snake):
            return adapt_backend_module(mod).require_symbol(snake)
    return None


def execute_op(
    cls: type | types.ModuleType | None,
    op_type: str,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager execute op operation on the active backend array.

    Args:
        cls (type | types.ModuleType | None): The class.
        op_type (str): Input `op_type` argument (op type).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        BackendNotSupportedError: If invalid inputs or unsupported runtime conditions trigger
            `BackendNotSupportedError`.
    """
    try:
        func_registry = mlx_eager_registry._registry.get(op_type)
        if func_registry is not None:
            return require_callable(func_registry)(mx, *args, **kwargs)
        if op_type not in ("Mul", "Sub", "Div"):
            func_registry = global_eager_registry.get(op_type)
            if func_registry is not None:
                return require_callable(func_registry)(mx, *args, **kwargs)
        func = _get_mlx_func(op_type)
        if func is None:
            raise AttributeError(f"No attribute found for {op_type}")
        return require_callable(func)(*args, **kwargs)
    except AttributeError:
        from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

        raise BackendNotSupportedError("Operation " + op_type + " is not implemented for the MLX backend.") from None


@mlx_eager_registry.register("Cast")
def _mlx_cast(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx cast operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tensor = args[0]
    dtype_val = kwargs.get("dtype") if "dtype" in kwargs else args[1]
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return tensor
    return adapt_backend_module(tensor).require_op("astype")(dtype)


@mlx_eager_registry.register("Relu")
def _mlx_relu(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Evaluate ReLU activation using MLX.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The MLX core module.
        x (BackendArg): Input tensor array.
        *args (BackendArg): Additional positional arguments.
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Rectified linear activated tensor array.
    """
    try:
        import mlx.nn as nn

        return require_callable(nn.relu)(x)
    except (ImportError, AttributeError):
        return adapt_backend_module(backend_module).require_symbol("maximum")(x, 0)


@mlx_eager_registry.register("RaggedTensorToDense")
def _mlx_ragged_tensor_to_dense(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    rt_input: BackendArg,
    default_value: BackendArg = None,
    **kwargs: BackendArg,
) -> BackendArg:
    """Convert a ragged tensor representation to a dense padded array using MLX.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The MLX backend module.
        rt_input (BackendArg): The input ragged tensor, sequence of arrays, or dict.
        default_value (BackendArg): The value used to pad the ragged dimensions.
        **kwargs (BackendArg): Additional keyword arguments.

    Returns:
        BackendArg: Dense padded array.
    """
    if (
        isinstance(rt_input, (list, tuple))
        and len(rt_input) > 0
        and adapt_backend_module(rt_input[0]).has_symbol("shape")
    ):
        elem_shapes = [coerce_shape(adapt_backend_module(x).require_symbol("shape")) for x in rt_input]
        max_len = max(int(sh[0]) if len(sh) > 0 else 1 for sh in elem_shapes)
        pad_val = default_value if default_value is not None else 0.0
        padded: list[BackendArg] = []
        for x, x_sh in zip(rt_input, elem_shapes):
            cur_len = int(x_sh[0]) if len(x_sh) > 0 else 1
            if cur_len < max_len:
                diff = max_len - cur_len
                pad_shape = (diff,) + tuple(int(d) for d in x_sh[1:]) if len(x_sh) > 1 else (diff,)
                pad_arr = adapt_backend_module(backend_module).require_symbol("full")(
                    pad_shape,
                    pad_val,
                    dtype=adapt_backend_module(x).get_symbol(
                        "dtype", adapt_backend_module(backend_module).require_symbol("float32")
                    ),
                )
                padded_x: BackendArg = adapt_backend_module(backend_module).require_symbol("concatenate")(
                    [x, pad_arr], axis=0
                )
            else:
                padded_x = x
            padded.append(padded_x)
        return adapt_backend_module(backend_module).require_symbol("stack")(padded, axis=0)

    values = None
    row_splits = None
    if isinstance(rt_input, dict) and "values" in rt_input and "row_splits" in rt_input:
        values = rt_input["values"]
        row_splits = rt_input["row_splits"]
    elif adapt_backend_module(rt_input).has_symbol("values") and adapt_backend_module(rt_input).has_symbol(
        "row_splits"
    ):
        values = adapt_backend_module(rt_input).require_symbol("values")
        row_splits = adapt_backend_module(rt_input).require_symbol("row_splits")

    if values is not None and row_splits is not None:
        splits_seq = coerce_operand_list(row_splits)
        num_rows = len(splits_seq) - 1
        splits_list = [
            int(
                coerce_int(
                    adapt_backend_module(s).require_op("item")() if adapt_backend_module(s).has_symbol("item") else s
                )
            )
            for s in splits_seq
        ]
        diffs = [splits_list[i + 1] - splits_list[i] for i in range(num_rows)]
        max_len = max(diffs) if diffs else 0
        pad_val = default_value if default_value is not None else 0.0
        val_shape = coerce_shape(adapt_backend_module(values).require_symbol("shape"))
        feature_shape = tuple(int(d) for d in val_shape[1:]) if len(val_shape) > 1 else ()
        dense_shape = (num_rows, max_len) + feature_shape
        dense_tensor = adapt_backend_module(backend_module).require_symbol("full")(
            dense_shape,
            pad_val,
            dtype=adapt_backend_module(values).get_symbol(
                "dtype", adapt_backend_module(backend_module).require_symbol("float32")
            ),
        )
        val_arr = coerce_array(backend_module, values)
        rows: list[BackendArg] = []
        for i in range(num_rows):
            start = splits_list[i]
            end = splits_list[i + 1]
            length = end - start
            row_val: BackendArg = val_arr[start:end]
            if length < max_len:
                pad_shape = (max_len - length,) + feature_shape
                pad_arr = adapt_backend_module(backend_module).require_symbol("full")(
                    pad_shape,
                    pad_val,
                    dtype=adapt_backend_module(values).get_symbol(
                        "dtype", adapt_backend_module(backend_module).require_symbol("float32")
                    ),
                )
                row_val = adapt_backend_module(backend_module).require_symbol("concatenate")([row_val, pad_arr], axis=0)
            rows.append(row_val)
        if rows:
            return adapt_backend_module(backend_module).require_symbol("stack")(rows, axis=0)
        return dense_tensor

    return rt_input


@mlx_eager_registry.register("TakeAlongAxis")
def _mlx_take_along_axis(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx take along axis operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("take_along_axis")(*args, **kwargs)


@mlx_eager_registry.register("Take")
def _mlx_take(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx take operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("take")(*args, **kwargs)


@mlx_eager_registry.register("TensorScatterUpdate")
def _mlx_tensor_scatter_update(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx tensor scatter update operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    (tensor, indices, updates) = (args[0], coerce_array(backend_module, args[1]), args[2])
    res = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx: tuple[EagerTensorProtocol, ...] = tuple(
        indices[..., dim]
        for dim in range(int(coerce_int(coerce_shape(adapt_backend_module(indices).require_symbol("shape"))[-1])))
    )
    res[idx] = updates
    return res


@mlx_eager_registry.register("TensorScatterAdd")
def _mlx_tensor_scatter_add(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement TensorScatterAdd.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    (tensor, indices, updates) = (args[0], coerce_array(backend_module, args[1]), args[2])
    res = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx: tuple[EagerTensorProtocol, ...] = tuple(
        indices[..., dim]
        for dim in range(int(coerce_int(coerce_shape(adapt_backend_module(indices).require_symbol("shape"))[-1])))
    )
    res[idx] = res[idx] + coerce_numeric(updates)
    return res


@mlx_eager_registry.register("TensorScatterMax")
def _mlx_tensor_scatter_max(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement TensorScatterMax.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    mx = backend_module

    (tensor, indices, updates) = (args[0], coerce_array(backend_module, args[1]), args[2])
    res = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx: tuple[EagerTensorProtocol, ...] = tuple(
        indices[..., dim]
        for dim in range(int(coerce_int(coerce_shape(adapt_backend_module(indices).require_symbol("shape"))[-1])))
    )
    res[idx] = adapt_backend_module(mx).require_symbol("maximum")(res[idx], updates)
    return res


@mlx_eager_registry.register("TensorScatterMin")
def _mlx_tensor_scatter_min(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement TensorScatterMin.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    mx = backend_module

    (tensor, indices, updates) = (args[0], coerce_array(backend_module, args[1]), args[2])
    res = adapt_backend_module(backend_module).require_symbol("array")(tensor)
    idx: tuple[EagerTensorProtocol, ...] = tuple(
        indices[..., dim]
        for dim in range(int(coerce_int(coerce_shape(adapt_backend_module(indices).require_symbol("shape"))[-1])))
    )
    res[idx] = adapt_backend_module(mx).require_symbol("minimum")(res[idx], updates)
    return res


@mlx_eager_registry.register("ScatterNd")
def _mlx_scatter_nd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx scatter nd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """

    @typing.runtime_checkable
    class _ScatterIndexProto(typing.Protocol):
        """Protocol for scatter index tensors supporting `.shape` and `__getitem__`."""

        @property
        def shape(self) -> BackendArg:
            """Return index tensor shape.

            Returns:
                BackendArg: Index shape sequence.
            """
            ...

        def __getitem__(self, key: BackendArg) -> EagerTensorProtocol:
            """Slice index tensor along coordinate dimensions.

            Args:
                key (BackendArg): Slice tuple.

            Returns:
                EagerTensorProtocol: Coordinate slice.
            """
            ...

    mod_adp = adapt_backend_module(backend_module)
    raw_indices = args[0]
    has_conv = mod_adp.has_symbol("asarray") or mod_adp.has_symbol("array")
    indices_val: BackendArg | _ScatterIndexProto = (
        mod_adp.asarray(raw_indices) if not isinstance(raw_indices, BackendArrayProtocol) and has_conv else raw_indices
    )
    assert isinstance(indices_val, _ScatterIndexProto)
    updates = args[1]
    shape = args[2] if len(args) > 2 else kwargs.get("shape")
    if adapt_backend_module(shape).has_symbol("data"):
        shape = adapt_backend_module(shape).require_symbol("data")
    if adapt_backend_module(shape).has_symbol("tolist"):
        shape = adapt_backend_module(shape).require_op("tolist")()
    if isinstance(shape, tuple):
        shape = list(shape)
    # Implement native MLX ScatterNd logic using array indexing
    res = mod_adp.require_symbol("zeros")(shape, dtype=adapt_backend_module(updates).require_symbol("dtype"))
    idx: tuple[EagerTensorProtocol, ...] = tuple(
        indices_val[(Ellipsis, dim)] for dim in range(int(coerce_int(coerce_shape(indices_val.shape)[-1])))
    )
    res[idx] = updates
    return res


@mlx_eager_registry.register("Reshape")
def _mlx_reshape(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx reshape operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[1] if len(args) > 1 else kwargs.get("newshape"))
    if adapt_backend_module(shape).has_symbol("data"):
        shape = adapt_backend_module(shape).require_symbol("data")
    if adapt_backend_module(shape).has_symbol("tolist"):
        shape = adapt_backend_module(shape).require_op("tolist")()
    if isinstance(shape, tuple):
        shape = list(shape)
    return adapt_backend_module(backend_module).require_symbol("reshape")(
        args[0] if "input" not in kwargs else kwargs["input"], shape
    )


_MLX_DTYPE_FALLBACK_MAP = {
    "bfloat16": "float32",
    "float8": "float32",
    "int4": "int8",
}


def _resolve_dtype(
    backend_module: BackendModuleProtocol | types.ModuleType | None, dtype_val: BackendArg
) -> BackendArg:
    """Resolve a given dtype object or string to a valid MLX dtype.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The MLX backend module.
        dtype_val (BackendArg): The requested dtype.

    Returns: mx.array: The resolved MLX dtype.
    """
    if dtype_val is None:
        return None
    dtype_str: str = str(dtype_val).split(".")[-1]
    dtype: BackendArg = adapt_backend_module(backend_module).get_symbol(dtype_str, dtype_val)
    if isinstance(dtype, str) and not adapt_backend_module(backend_module).has_symbol(dtype_str):
        for k, v in _MLX_DTYPE_FALLBACK_MAP.items():
            if k in dtype:
                return adapt_backend_module(backend_module).require_symbol(v)
    return dtype


@mlx_eager_registry.register("Zeros")
def _mlx_zeros(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx zeros operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if adapt_backend_module(shape).has_symbol("data"):
        shape = adapt_backend_module(shape).require_symbol("data")
    dtype_val = kwargs.get("dtype", None)
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return adapt_backend_module(backend_module).require_symbol("zeros")(shape)
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return adapt_backend_module(backend_module).require_symbol("zeros")(shape, dtype=dtype)
    except TypeError:
        return adapt_backend_module(backend_module).require_symbol("zeros")(shape)


@mlx_eager_registry.register("Ones")
def _mlx_ones(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx ones operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if adapt_backend_module(shape).has_symbol("data"):
        shape = adapt_backend_module(shape).require_symbol("data")
    dtype_val = kwargs.get("dtype", adapt_backend_module(backend_module).get_symbol("float32", None))
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return adapt_backend_module(backend_module).require_symbol("ones")(shape)
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return adapt_backend_module(backend_module).require_symbol("ones")(shape, dtype=dtype)
    except TypeError:
        return adapt_backend_module(backend_module).require_symbol("ones")(shape)


@mlx_eager_registry.register("Full")
def _mlx_full(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx full operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if adapt_backend_module(shape).has_symbol("data"):
        shape = adapt_backend_module(shape).require_symbol("data")
    fill_value = kwargs.get("fill_value", args[1] if len(args) > 1 else 0)
    dtype_val = kwargs.get("dtype", adapt_backend_module(backend_module).get_symbol("float32", None))
    dtype = _resolve_dtype(backend_module, dtype_val)
    if dtype is None:
        return adapt_backend_module(backend_module).require_symbol("full")(shape, fill_value)
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return adapt_backend_module(backend_module).require_symbol("full")(shape, fill_value, dtype=dtype)
    except TypeError:
        return adapt_backend_module(backend_module).require_symbol("full")(shape, fill_value)


@mlx_eager_registry.register("Partition")
def _parse_partition_k(k: BackendArg) -> int:
    """Parse k parameter for partition.

    Args:
        k (BackendArg): Input `k` argument (k).

    Returns:
        int: Computed `int` output.
    """
    if adapt_backend_module(k).has_symbol("item"):
        return int(coerce_int(adapt_backend_module(k).require_op("item")()))
    if adapt_backend_module(k).has_symbol("data") and adapt_backend_module(
        adapt_backend_module(k).require_symbol("data")
    ).has_symbol("item"):
        return int(
            coerce_int(adapt_backend_module(adapt_backend_module(k).require_symbol("data")).require_op("item")())
        )
    return int(coerce_int(k))


def _mlx_partition(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx partition operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    a = args[0]
    k_raw = kwargs.get("k", args[1] if len(args) > 1 else 1)
    k: int = int(_parse_partition_k(k_raw))
    return_indices = kwargs.get("return_indices", None)
    kth: int = max(0, int(coerce_int(coerce_shape(adapt_backend_module(a).require_symbol("shape"))[-1])) - k)
    if return_indices is False:
        if adapt_backend_module(backend_module).has_symbol("topk"):
            return adapt_backend_module(backend_module).require_symbol("topk")(a, k)
        return adapt_backend_module(backend_module).require_symbol("partition")(a, kth, axis=-1)[..., -k:]
    indices = adapt_backend_module(backend_module).require_symbol("argpartition")(a, kth, axis=-1)[..., -k:]
    if return_indices is True:
        return indices
    values = adapt_backend_module(backend_module).require_symbol("take_along_axis")(a, indices, axis=-1)
    return (values, indices)


@mlx_eager_registry.register("NanToNum")
def _mlx_nan_to_num(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx nan to num operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    valid_kwargs: dict[str, BackendArg] = {}
    for key in ("nan", "posinf", "neginf"):
        if key in kwargs:
            raw_val: BackendArg = kwargs[key]
            if adapt_backend_module(raw_val).has_symbol("item"):
                valid_kwargs[key] = float(coerce_float(adapt_backend_module(raw_val).require_op("item")()))
            elif adapt_backend_module(raw_val).has_symbol("data") and adapt_backend_module(
                adapt_backend_module(raw_val).require_symbol("data")
            ).has_symbol("item"):
                valid_kwargs[key] = float(
                    coerce_float(
                        adapt_backend_module(adapt_backend_module(raw_val).require_symbol("data")).require_op("item")()
                    )
                )
            elif raw_val is not None:
                valid_kwargs[key] = float(coerce_float(raw_val))
            else:
                valid_kwargs[key] = None
    return adapt_backend_module(backend_module).require_symbol("nan_to_num")(*args, **valid_kwargs)


@mlx_eager_registry.register("Cummax")
def _mlx_cummax(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx cummax operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    dtype: BackendArg = kwargs.pop("dtype", None)
    res: BackendArg = adapt_backend_module(backend_module).require_symbol("cummax")(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        dt_val = adapt_backend_module(dtype).get_symbol("value", dtype)
        res = adapt_backend_module(res).require_op("astype")(
            adapt_backend_module(backend_module).get_symbol(str(dt_val), dt_val)
        )
    return res


@mlx_eager_registry.register("Cummin")
def _mlx_cummin(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx cummin operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    dtype: BackendArg = kwargs.pop("dtype", None)
    res: BackendArg = adapt_backend_module(backend_module).require_symbol("cummin")(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        dt_val = adapt_backend_module(dtype).get_symbol("value", dtype)
        res = adapt_backend_module(res).require_op("astype")(
            adapt_backend_module(backend_module).get_symbol(str(dt_val), dt_val)
        )
    return res


@mlx_eager_registry.register("Cumprod")
def _mlx_cumprod(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx cumprod operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    dtype: BackendArg = kwargs.pop("dtype", None)
    res: BackendArg = adapt_backend_module(backend_module).require_symbol("cumprod")(*args, **kwargs)
    if dtype is not None and str(dtype) != "None":
        dt_val = adapt_backend_module(dtype).get_symbol("value", dtype)
        res = adapt_backend_module(res).require_op("astype")(
            adapt_backend_module(backend_module).get_symbol(str(dt_val), dt_val)
        )
    return res


@mlx_eager_registry.register("Slice")
def _mlx_slice(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx slice operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    a = coerce_array(backend_module, args[0])
    dim: int = int(coerce_int(kwargs.get("dim", 0)))
    start = coerce_opt_int(kwargs.get("start"))
    end = coerce_opt_int(kwargs.get("end"))
    step = coerce_opt_int(kwargs.get("step", 1))
    sl: list[builtins.slice[int | None, int | None, int | None]] = [builtins.slice(None)] * len(
        coerce_shape(adapt_backend_module(a).require_symbol("shape"))
    )
    sl[dim] = builtins.slice(start, end, step)
    return a[tuple(sl)]


@mlx_eager_registry.register("Eye")
def _mlx_eye(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx eye operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    n_arg = args[0]
    if adapt_backend_module(n_arg).has_symbol("data"):
        n_arg = adapt_backend_module(n_arg).require_symbol("data")
    m_arg = args[1] if len(args) > 1 else None
    if adapt_backend_module(m_arg).has_symbol("data"):
        m_arg = adapt_backend_module(m_arg).require_symbol("data")
    m_val: float = int(coerce_int(m_arg)) if m_arg is not None else int(coerce_int(n_arg))
    k_val: float = int(coerce_int(kwargs.get("k", 0)))
    return adapt_backend_module(backend_module).require_symbol("eye")(
        n=int(coerce_int(n_arg)),
        m=m_val,
        k=k_val,
        dtype=adapt_backend_module(backend_module).require_symbol(coerce_str(kwargs.get("dtype", "float32"))),
    )


@mlx_eager_registry.register("Rope")
def _mlx_rope(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Apply Rotary Positional Encoding using MLX.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    dim = kwargs.get("dim")
    base: float = coerce_float(kwargs.get("base", 10000.0))
    offset: int = coerce_int(kwargs.get("offset", 0))
    traditional: bool = coerce_bool(kwargs.get("traditional", False))
    scale: float = coerce_float(kwargs.get("scale", 1.0))
    return (
        adapt_backend_module(mx)
        .require_symbol("fast")
        .rope(x, dim, traditional=traditional, base=base, scale=scale, offset=offset)
    )


@mlx_eager_registry.register("Variance")
def _mlx_variance(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mlx variance operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    kwargs.setdefault("ddof", 0)
    return adapt_backend_module(backend_module).require_symbol("var")(*args, **kwargs)


@typing.runtime_checkable
class _MlxDistributedNamespaceProtocol(typing.Protocol):
    """Protocol for MLX modules exposing a `distributed` namespace."""

    distributed: EagerTensorProtocol


@mlx_eager_registry.register("AllReduce")
def _mlx_all_reduce(
    backend_module: BackendModuleProtocol | types.ModuleType | None, tensor: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement AllReduce for MLX eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The MLX backend module.
        tensor (BackendArg): The input tensor.
        **kwargs (BackendArg): Keyword args.

    Returns: mx.array: The reduced tensor.
    """
    mx = backend_module

    dist_mod = adapt_backend_module(mx).get_symbol("distributed", None)
    if dist_mod is not None and adapt_backend_module(dist_mod).has_symbol("all_sum"):
        return adapt_backend_module(mx).require_symbol("distributed").all_sum(tensor)
    return tensor


@mlx_eager_registry.register("AllGather")
def _mlx_all_gather(
    backend_module: BackendModuleProtocol | types.ModuleType | None, tensor: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement AllGather for MLX eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The MLX backend module.
        tensor (BackendArg): The input tensor.
        **kwargs (BackendArg): Keyword args.

    Returns: mx.array: The gathered tensor.
    """
    mx = backend_module

    dist_mod = adapt_backend_module(mx).get_symbol("distributed", None)
    if dist_mod is not None and adapt_backend_module(dist_mod).has_symbol("all_gather"):
        return adapt_backend_module(mx).require_symbol("distributed").all_gather(tensor)
    return adapt_backend_module(mx).require_symbol("expand_dims")(tensor, axis=0)


@mlx_eager_registry.register("AllToAll")
def _mlx_all_to_all(
    backend_module: BackendModuleProtocol | types.ModuleType | None, tensor: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement AllToAll for MLX eager mode.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The MLX backend module.
        tensor (BackendArg): The input tensor.
        **kwargs (BackendArg): Keyword args.

    Returns: mx.array: The distributed tensor.
    """
    mx = backend_module

    dist_mod = adapt_backend_module(mx).get_symbol("distributed", None)
    if dist_mod is not None and adapt_backend_module(dist_mod).has_symbol("all_to_all"):
        return adapt_backend_module(mx).require_symbol("distributed").all_to_all(tensor)
    return tensor


@typing.runtime_checkable
class _SliceableChunkProto(typing.Protocol):
    """Protocol for tensor/array objects supporting `.shape`, `.ndim`, and tuple slice indexing."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape tuple.

        Returns:
            tuple[int, ...]: Array dimensions.
        """
        ...

    @property
    def ndim(self) -> int:
        """Return array rank.

        Returns:
            int: Number of dimensions.
        """
        ...

    def __getitem__(
        self,
        index: tuple[builtins.slice[int | None, int | None, int | None], ...],
    ) -> BackendArg:
        """Slice array along dimensions.

        Args:
            index (tuple[builtins.slice[int | None, int | None, int | None], ...]): Tuple of slices.

        Returns:
            BackendArg: Sliced array chunk.
        """
        ...


def _as_sliceable_chunk(mod: BackendArg, val: BackendArg) -> _SliceableChunkProto | EagerTensorProtocol:
    """Coerce `val` via `mod` if supported, falling back to `_SliceableChunkProto` when `val` is already sliceable.

    Args:
        mod (BackendArg): MLX backend module or None.
        val (BackendArg): Candidate array or tensor stub.

    Returns:
        _SliceableChunkProto | EagerTensorProtocol: Sliceable array object.
    """
    try:
        return coerce_array(mod, val)
    except (ValueError, TypeError, AttributeError):
        assert isinstance(val, _SliceableChunkProto)
        return val


def _slice_scatter_chunk(tensor: BackendArg, scatter_dim: int, rank: int, world_size: int) -> BackendArg:
    """Slice a tensor along scatter_dim for a given rank and world_size.

    Args:
        tensor (BackendArg): Input tensor to slice.
        scatter_dim (int): Dimension along which to partition.
        rank (int): Current rank index.
        world_size (int): Total number of ranks.

    Returns:
        BackendArg: Sliced chunk for the specified rank.
    """
    t_arr = _as_sliceable_chunk(mx, tensor)
    dim_size = int(t_arr.shape[scatter_dim])
    chunk_size = dim_size // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    slices: list[builtins.slice[int | None, int | None, int | None]] = [builtins.slice(None)] * t_arr.ndim
    slices[scatter_dim] = builtins.slice(start_idx, end_idx)
    return t_arr[tuple(slices)]


def _mlx_native_reduce_scatter(mx_mod: BackendArg, tensor: BackendArg, kwargs: dict[str, BackendArg]) -> BackendArg:
    """Execute native MLX distributed all_sum followed by rank chunk slicing.

    Args:
        mx_mod (BackendArg): Loaded MLX module with distributed support.
        tensor (BackendArg): Input tensor.
        kwargs (dict[str, BackendArg]): Distributed options (group, scatter_dim).

    Returns:
        BackendArg: Reduced and scattered tensor slice.
    """
    dist_mod = adapt_backend_module(mx_mod).require_symbol("distributed")
    dist_adp = adapt_backend_module(dist_mod)
    reduced = dist_adp.require_op("all_sum")(tensor)
    raw_group = kwargs.get("group")
    group = raw_group if raw_group is not None else None
    try:
        rank = (
            coerce_int(adapt_backend_module(group).require_op("rank")())
            if group is not None
            else (
                coerce_int(adapt_backend_module(dist_adp.require_op("init")()).require_op("rank")())
                if dist_adp.has_symbol("init")
                else 0
            )
        )
        world_size = (
            coerce_int(adapt_backend_module(group).require_op("size")())
            if group is not None
            else (
                coerce_int(adapt_backend_module(dist_adp.require_op("init")()).require_op("size")())
                if dist_adp.has_symbol("init")
                else 1
            )
        )
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
        rank = 0
        world_size = 1

    scatter_dim = int(coerce_int(kwargs.get("scatter_dim", 0)))
    return _slice_scatter_chunk(reduced, scatter_dim, rank, world_size)


def _combine_ring_chunks(mx_mod: BackendArg, accum: BackendArg, peer_chunk: BackendArg, red_op: str) -> BackendArg:
    """Combine two tensor chunks using the specified reduction operator.

    Args:
        mx_mod (BackendArg): Loaded MLX module or None.
        accum (BackendArg): Running accumulator chunk.
        peer_chunk (BackendArg): Incoming peer chunk.
        red_op (str): Uppercase reduction operation name.

    Returns:
        BackendArg: Combined chunk result.
    """
    mod = mx_mod if mx_mod is not None else None
    mod_adp = adapt_backend_module(mod)
    op_map = {"PROD": "multiply", "PRODUCT": "multiply", "MUL": "multiply", "MAX": "maximum", "MIN": "minimum"}
    mlx_op = op_map.get(red_op, "add")
    if mod is not None and mod_adp.has_symbol(mlx_op):
        return mod_adp.require_op(mlx_op)(accum, peer_chunk)
    if red_op in ("PROD", "PRODUCT", "MUL"):
        return (
            accum * peer_chunk
            if isinstance(accum, BackendArrayProtocol) and isinstance(peer_chunk, BackendArrayProtocol)
            else coerce_numeric(accum) * coerce_numeric(peer_chunk)
        )
    if red_op == "MAX":
        return accum if coerce_float(accum) >= coerce_float(peer_chunk) else peer_chunk
    if red_op == "MIN":
        return accum if coerce_float(accum) <= coerce_float(peer_chunk) else peer_chunk
    return (
        accum + peer_chunk
        if isinstance(accum, BackendArrayProtocol) and isinstance(peer_chunk, BackendArrayProtocol)
        else coerce_numeric(accum) + coerce_numeric(peer_chunk)
    )


def _mlx_ring_reduce_scatter(
    mx_mod: BackendArg,
    all_ranks_data: list[BackendArg] | tuple[BackendArg, ...],
    scatter_dim: int,
    rank: int,
    red_op: str,
) -> BackendArg:
    """Perform host-level ring reduce-scatter across all_ranks_data.

    Args:
        mx_mod (BackendArg): Loaded MLX module or None.
        all_ranks_data (list[BackendArg] | tuple[BackendArg, ...]): Per-rank input tensors.
        scatter_dim (int): Dimension to scatter along.
        rank (int): Current rank index.
        red_op (str): Uppercase reduction operator string.

    Returns:
        BackendArg: Accumulated chunk for the current rank.
    """
    num_ranks = len(all_ranks_data)
    dim_size = int(
        coerce_int(coerce_shape(adapt_backend_module(all_ranks_data[0]).get_symbol("shape", (1,)))[scatter_dim])
    )
    chunk_size = dim_size // num_ranks
    start = rank * chunk_size
    end = start + chunk_size
    slices: list[builtins.slice[int | None, int | None, int | None]] = [builtins.slice(None)] * int(
        coerce_int(adapt_backend_module(all_ranks_data[0]).get_symbol("ndim", 1))
    )
    slices[scatter_dim] = builtins.slice(start, end)
    sl_tuple = tuple(slices)

    accum: BackendArg = _as_sliceable_chunk(mx_mod, all_ranks_data[rank])[sl_tuple]
    for step in range(1, num_ranks):
        peer = (rank + step) % num_ranks
        peer_chunk = _as_sliceable_chunk(mx_mod, all_ranks_data[peer])[sl_tuple]
        accum = _combine_ring_chunks(mx_mod, accum, peer_chunk, red_op)
    return accum


@mlx_eager_registry.register("ReduceScatter")
def _mlx_reduce_scatter(
    backend_module: BackendModuleProtocol | types.ModuleType | None, tensor: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement ReduceScatter for MLX eager mode using native collectives or ring reduce-scatter fallback.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The MLX backend module.
        tensor (BackendArg): The input tensor.
        **kwargs (BackendArg): Keyword args including op_type, scatter_dim, rank, world_size, all_ranks_data.

    Returns:
        BackendArg: The scattered reduced tensor.

    Raises:
        BackendNotSupportedError: When raised during `_mlx_reduce_scatter` (ReduceScatter is not supported in the
            current MLX version).
    """
    mx_mod = backend_module

    dist_mod = adapt_backend_module(mx_mod).get_symbol("distributed", None) if mx_mod is not None else None
    if mx_mod is not None and dist_mod is not None and adapt_backend_module(dist_mod).has_symbol("all_sum"):
        return _mlx_native_reduce_scatter(mx_mod, tensor, kwargs)

    all_ranks_data = kwargs.get("all_ranks_data")
    scatter_dim = int(coerce_int(kwargs.get("scatter_dim", 0)))
    red_op = str(kwargs.get("op_type", "SUM")).upper()
    rank = int(coerce_int(kwargs.get("rank", 0)))

    if all_ranks_data:
        return _mlx_ring_reduce_scatter(mx_mod, coerce_operand_list(all_ranks_data), scatter_dim, rank, red_op)

    world_size = int(coerce_int(kwargs.get("world_size", 1)))
    if adapt_backend_module(tensor).has_symbol("shape"):
        if world_size <= 1:
            _typed_ret_1: BackendArg = tensor
            return _typed_ret_1
        return _slice_scatter_chunk(tensor, scatter_dim, rank, world_size)

    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    raise BackendNotSupportedError("ReduceScatter is not supported in the current MLX version.")
