"""Module state.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Generate random operations."""
import uuid

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
    inputs: list[Tensor],
    shape: tuple[int, ...],
    dtype: dtypes.DType,
    attributes: dict[str, object] | None = None,
) -> Tensor:
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
        attrs: object = dict(attributes) if attributes is not None else {}
        if "shape" not in attrs:
            attrs["shape"] = shape
        if "dtype" not in attrs:
            attrs["dtype"] = getattr(dtype, "value", dtype)
        res: object = _dispatch_random_eager(op_type.lower(), op_type, *inputs, **attrs)
        return Tensor(res, TensorConfig(shape, dtype, config.default_device))

    out_id: object = str(uuid.uuid4())
    node: object = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[getattr(inp.data, "id", str(id(inp))) for inp in inputs],
        attributes=attributes or {},
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)
    proxy: object = ProxyTensor(id=out_id, shape=shape, dtype=getattr(dtype, "value", str(dtype)))
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


def _dispatch_random_eager(func_name: str, op_name: str, *args: object, **kwargs: object) -> object:
    """Help to dispatch random functions in eager mode.

    Args:
        func_name (str): The func_name parameter.
        op_name (str): The op_name parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    backend: object = get_active_backend()
    return backend.execute_op(op_name, *args, **kwargs)


def _dispatch_random(func_name: str, *args: object, **kwargs: object) -> object:
    """Evaluate _dispatch_random operation.

    Args:
        func_name (str): The func_name parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    op_name: object = "".join(word.capitalize() for word in func_name.split("_"))
    if config.eager_mode:
        return _dispatch_random_eager(func_name, op_name, *args, **kwargs)

    op_cls: object = get_op(op_name)
    if op_cls:
        return op_cls()(*args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(op_name, list(args), kwargs, (), "float32")


def rng_uniform(a: object, b: object, shape: object, dtype: object = None) -> object:
    """Generate uniform random values.

    Args:
        a (object): Lower bound.
        b (object): Upper bound.
        shape (object): Shape.
        dtype (object): Data type.

    Returns: object: Random values.
    """
    return _dispatch_random("rng_uniform", a, b, shape=shape, dtype=dtype)


def _get_numpy_rng(*args: object, **kwargs: object) -> object:
    """Get the NumPy RNG instance from the numpy backend.

    Args:
        *args (object): Positional arguments.
        **kwargs (object): Keyword arguments.

    Returns: object: The NumPy RNG instance.
    """
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    backend_cls: object = BackendRegistry.get("numpy")
    return backend_cls.get_numpy_rng(*args, **kwargs)
