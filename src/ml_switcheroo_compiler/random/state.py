from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Generate random operations."""
import uuid
from typing import Any

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

# In tracing mode, we need to map to OpDefs just like core tensor ops do through get_op
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

# It's an OpDef class


def _emit_random_node(
    op_type: str,
    inputs: list[Tensor],
    shape: tuple[int, ...],
    dtype: dtypes.DType,
    attributes: dict | None = None,
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
        attrs = dict(attributes) if attributes is not None else {}
        if "shape" not in attrs:
            attrs["shape"] = shape
        if "dtype" not in attrs:
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
    proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


def PRNGKey(seed: int) -> Tensor:
    """Create a PRNG key given an integer seed.

    Args:
        seed (int): The random seed.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if config.eager_mode:
        res = _dispatch_random_eager("prng_key", "PRNGKey", seed)
        return Tensor(res, TensorConfig((2,), dtypes.DType.UInt32, config.default_device))

    # Trace as a creation node
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="PRNGKey",
        inputs=[],
        attributes={"seed": seed},
        shape_metadata=(2,),
    )
    if global_tracing_state.is_tracing:
        global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=(2,), dtype="uint32")
    return Tensor(proxy, TensorConfig((2,), dtypes.DType.UInt32, config.default_device))


def split(key: Tensor, num: int = 2) -> Tensor:
    """Split a PRNG key into num new keys.

    Args:
        key (Tensor): The PRNG key.
        num (int): The num parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if config.eager_mode:
        res = _dispatch_random_eager("split", "RandomSplit", key.data, num=num)
        return Tensor(res, TensorConfig((num, 2), dtypes.DType.UInt32, config.default_device))
    return _emit_random_node("RandomSplit", [key], (num, 2), dtypes.DType.UInt32, {"num": num})


def fold_in(key: Tensor, data: int) -> Tensor:
    """Folds in data to a PRNG key to derive a new key.

    Args:
        key (Tensor): The PRNG key.
        data (int): The data parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if config.eager_mode:
        res = _dispatch_random_eager("fold_in", "RandomFoldIn", key.data, data=data)
        return Tensor(res, TensorConfig((2,), dtypes.DType.UInt32, config.default_device))
    return _emit_random_node("RandomFoldIn", [key], (2,), dtypes.DType.UInt32, {"data": data})


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


def key(*args: Any, **kwargs: Any) -> Any:
    """Evaluate key operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _dispatch_random("key", *args, **kwargs)


def key_data(*args: Any, **kwargs: Any) -> Any:
    """Evaluate key_data operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _dispatch_random("key_data", *args, **kwargs)


def key_impl(*args: Any, **kwargs: Any) -> Any:
    """Evaluate key_impl operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _dispatch_random("key_impl", *args, **kwargs)


def wrap_key_data(*args: Any, **kwargs: Any) -> Any:
    """Evaluate wrap_key_data operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _dispatch_random("wrap_key_data", *args, **kwargs)


def clone(*args: Any, **kwargs: Any) -> Any:
    """Evaluate clone operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _dispatch_random("clone", *args, **kwargs)


def bits(*args: Any, **kwargs: Any) -> Any:
    """Evaluate bits operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _dispatch_random("bits", *args, **kwargs)


def rng_bit_generator(key: Any, shape: Any, dtype: Any = None) -> Any:
    """Generate random bits.

    Args:
        key (object): PRNGKey.
        shape (object): Shape.
        dtype (object): Data type.

    Returns: Any: Random bits.
    """
    return _dispatch_random("rng_bit_generator", key, shape=shape, dtype=dtype)


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
