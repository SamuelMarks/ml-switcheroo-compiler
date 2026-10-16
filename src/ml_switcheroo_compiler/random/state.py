"""Random state and dispatch primitives."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Union

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.tracing import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

RandomArg = Union[Tensor, str, float, int, bool, Sequence[int], tuple[int, ...], dtypes.DType, None]


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
    dtype_str = getattr(dtype, "value") if hasattr(dtype, "value") else (str(dtype) if dtype is not None else "float32")
    if config.eager_mode:
        eager_kwargs: dict[str, RandomArg] = dict(attributes) if attributes else {}
        if "shape" not in eager_kwargs:
            eager_kwargs["shape"] = shape_tuple
        if "dtype" not in eager_kwargs:
            eager_kwargs["dtype"] = dtype_str
        res = _dispatch_random_eager(op_type.lower(), op_type, *inputs, **eager_kwargs)
        res_shape = getattr(res, "shape", shape_tuple)
        return Tensor(res, TensorConfig(res_shape, dtype, "cpu"))

    out_id = str(uuid.uuid4())
    in_ids = [getattr(inp.data, "id") if hasattr(inp, "data") and hasattr(inp.data, "id") else str(inp) for inp in inputs]
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
    from typing import cast

    del func_name
    return cast(TensorLike, get_active_backend().execute_op(op_name, *args, **kwargs))


def _dispatch_random(func_name: str, *args: RandomArg, **kwargs: RandomArg) -> Tensor:
    """Dispatch random operation in eager or tracing mode.

    Args:
        func_name (str): Random function name.
        *args (RandomArg): Positional arguments.
        **kwargs (RandomArg): Keyword arguments.

    Returns:
        Tensor: Resulting tensor.
    """
    from typing import cast

    op_name = "".join(word.capitalize() for word in func_name.split("_"))
    if config.eager_mode:
        return cast(Tensor, _dispatch_random_eager(func_name, op_name, *args, **kwargs))
    op_cls = get_op(op_name)
    if op_cls is not None:
        return op_cls()(*args, **kwargs)
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
