"""Random state and dispatch primitives."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.protocols import has_backend_member
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar, is_traced_data
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.ops import get_op
from ml_switcheroo_compiler.tracing import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

RandomArg = typing.Union[Tensor, str, float, int, bool, Sequence[int], tuple[int, ...], dtypes.DType, "None"]


class _DataWrapperProtocol(typing.Protocol):
    """Protocol for tensor wrappers exposing a `.data` attribute."""

    data: TensorOrScalar


def _is_data_wrapper(val: TensorOrScalar) -> typing.TypeGuard[_DataWrapperProtocol]:
    """Check whether `val` exposes a `.data` attribute.

    Args:
        val (TensorOrScalar): Candidate input operand.

    Returns:
        bool: True if `val` has `.data`.
    """
    return has_backend_member(val, "data")


def _is_random_tensor_like(val: TensorOrScalar) -> typing.TypeGuard[TensorLike]:
    """Type guard narrowing `val` to `TensorLike`.

    Args:
        val (TensorOrScalar): Evaluated backend value.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def _is_random_tensor(val: TensorOrScalar | Tensor) -> typing.TypeGuard[Tensor]:
    """Type guard narrowing `val` to `Tensor`.

    Args:
        val (TensorOrScalar | Tensor): Candidate value.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def _ensure_tensor_obj(val: TensorOrScalar | Tensor) -> Tensor:
    """Wrap `val` in a `Tensor` if it is not already a `Tensor`.

    Args:
        val (TensorOrScalar | Tensor): Tensor or backend array.

    Returns:
        Tensor: Wrapped or narrowed `Tensor` instance.
    """
    assert _is_random_tensor(val)
    return val


def _emit_random_node(
    op_type: str,
    inputs: Sequence[Tensor | str | float | int],
    shape: Sequence[int] | tuple[int, ...] | int,
    dtype: dtypes.DType | str | None,
    attributes: dict[str, str | int | float | bool | tuple[int, ...] | None] | None = None,
) -> Tensor:
    """Emit a symbolic or eager random operation node.

    Args:
        op_type (str): Canonical operation type name.
        inputs (Sequence[Tensor | str | float | int]): Input operands.
        shape (Sequence[int] | tuple[int, ...] | int): Output shape specification.
        dtype (dtypes.DType | str | None): Output data type.
        attributes (dict[str, str | int | float | bool | tuple[int, ...] | None] | None): Optional node attributes.

    Returns:
        Tensor: Resulting tensor.
    """
    shape_tuple: tuple[int, ...] = (shape,) if isinstance(shape, int) else tuple(shape)
    dtype_str = dtype.value if isinstance(dtype, dtypes.DType) else (str(dtype) if dtype is not None else "float32")
    if config.eager_mode:
        eager_kwargs: dict[str, RandomArg] = dict(attributes) if attributes else {}
        if "shape" not in eager_kwargs:
            eager_kwargs["shape"] = shape_tuple
        if "dtype" not in eager_kwargs:
            eager_kwargs["dtype"] = dtype_str
        res = _dispatch_random_eager(op_type.lower(), op_type, *inputs, **eager_kwargs)
        return Tensor(res, TensorConfig(shape_tuple, dtype, "cpu"))

    out_id = str(uuid.uuid4())
    in_ids = [str(inp.data.id) if _is_data_wrapper(inp) and is_traced_data(inp.data) else str(inp) for inp in inputs]
    attrs = dict(attributes) if attributes is not None else {}

    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=in_ids,
        attributes=attrs,
        shape_metadata=shape_tuple,
    )
    global_tracing_state.add_node(node)
    return Tensor(ProxyTensor(id=out_id, shape=shape_tuple, dtype=dtype_str), TensorConfig(shape_tuple, dtype, "cpu"))


def _dispatch_random_eager(func_name: str, op_name: str, *args: RandomArg, **kwargs: RandomArg) -> TensorLike:
    """Dispatch random operation in eager mode.

    Args:
        func_name (str): Random function name.
        op_name (str): Operation PascalCase identifier.
        *args (RandomArg): Positional arguments.
        **kwargs (RandomArg): Keyword arguments.

    Returns:
        TensorLike: Evaluated backend array.
    """
    del func_name
    out = get_active_backend().execute_op(op_name, *args, **kwargs)
    assert _is_random_tensor_like(out)
    return out


def _dispatch_random(func_name: str, *args: RandomArg, **kwargs: RandomArg) -> Tensor:
    """Dispatch random operation in eager or tracing mode.

    Args:
        func_name (str): Random function name.
        *args (RandomArg): Positional arguments.
        **kwargs (RandomArg): Keyword arguments.

    Returns:
        Tensor: Resulting tensor.
    """
    from ml_switcheroo_compiler.ops.base import OpDef

    op_name = "".join(word.capitalize() for word in func_name.split("_"))
    if config.eager_mode:
        return _ensure_tensor_obj(_dispatch_random_eager(func_name, op_name, *args, **kwargs))
    raw_op_cls: type[OpDef] | None = get_op(op_name)
    if raw_op_cls is not None:
        return _ensure_tensor_obj(raw_op_cls()(*args, **kwargs))
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(op_name, list(args), dict(kwargs), (), "float32")


def rng_uniform(
    a: float | int,
    b: float | int,
    shape: Sequence[int] | int,
    dtype: dtypes.DType | str | None = None,
) -> Tensor:
    """Sample uniform values in [a, b).

    Args:
        a (float | int): Lower bound.
        b (float | int): Upper bound.
        shape (Sequence[int] | int): Target shape.
        dtype (dtypes.DType | str | None): Optional data type.

    Returns:
        Tensor: Uniformly sampled tensor.
    """
    return _dispatch_random("rng_uniform", a, b, shape=shape, dtype=dtype)


def _get_numpy_rng(*args: int | None, **kwargs: int | None) -> TensorLike:
    """Obtain a NumPy random number generator instance via BackendRegistry.

    Args:
        *args (int | None): Positional seed arguments.
        **kwargs (int | None): Keyword seed arguments.

    Returns:
        TensorLike: NumPy RNG instance.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    backend_cls = BackendRegistry.get("numpy")
    return backend_cls.get_numpy_rng(*args, **kwargs)
