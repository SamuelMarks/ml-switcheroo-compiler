"""Module state.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Generate random operations."""
import uuid
from typing import Any

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def _emit_random_node(
    op_type: str,
    inputs: list[Tensor],  # type: ignore
    shape: tuple[int, ...],
    dtype: dtypes.DType,
    attributes: dict[str, Any] | None = None,
) -> Tensor:  # type: ignore
    """Evaluate _emit_random_node operation.

    Args:
        op_type (str): The op_type parameter.
        inputs (object): The inputs parameter.
        shape (object): The shape parameter.
        dtype (object): The dtype parameter.
        attributes (object): The attributes parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        attrs = dict(attributes) if attributes is not None else {}
        if "shape" not in attrs:  # pragma: no branch
            attrs["shape"] = shape
        if "dtype" not in attrs:  # pragma: no branch
            attrs["dtype"] = getattr(dtype, "value", dtype)
        res = _dispatch_random_eager(op_type.lower(), op_type, *inputs, **attrs)
        return Tensor(res, TensorConfig(shape, dtype, config.default_device))

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        attributes=attributes or {},
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=shape, dtype=getattr(dtype, "value", str(dtype)))  # type: ignore
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


def _dispatch_random_eager(func_name: str, op_name: str, *args: Any, **kwargs: Any) -> Any:
    """Help to dispatch random functions in eager mode.

    Args:
        func_name (str): The func_name parameter.
        op_name (str): The op_name parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    backend = get_active_backend()
    return backend.execute_op(op_name, *args, **kwargs)


def _dispatch_random(func_name: str, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _dispatch_random operation.

    Args:
        func_name (str): The func_name parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    op_name = "".join(word.capitalize() for word in func_name.split("_"))
    if config.eager_mode:
        return _dispatch_random_eager(func_name, op_name, *args, **kwargs)

    op_cls = get_op(op_name)
    if op_cls:
        return op_cls()(*args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(op_name, list(args), kwargs, (), "float32")


def rng_uniform(a: Any, b: Any, shape: Any, dtype: Any = None) -> Any:
    """Generate uniform random values.

    Args:
        a (object): Lower bound.
        b (object): Upper bound.
        shape (object): Shape.
        dtype (object): Data type.

    Returns: Any: Random values.
    """
    return _dispatch_random("rng_uniform", a, b, shape=shape, dtype=dtype)


def _get_numpy_rng(*args: Any, **kwargs: Any) -> Any:
    """Get the NumPy RNG instance from the numpy backend.

    Args:
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns: Any: The NumPy RNG instance.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    backend_cls = BackendRegistry.get("numpy")
    return backend_cls.get_numpy_rng(*args, **kwargs)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
