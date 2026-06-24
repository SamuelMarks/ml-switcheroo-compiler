"""Backend utilities."""

import mlx.core as mx

from ml_switcheroo_compiler.backends.eager import execute_generic_op
from ml_switcheroo_compiler.backends.eager_registry import mlx_eager_registry


def _to_numpy(val: object) -> object:
    """Function docstring.

    Args:
        val: Arg.
    """
    import numpy as np

    if isinstance(val, mx.array):
        return np.array(val)
    return val


def _from_numpy(val: object) -> object:
    """Function docstring.

    Args:
        val: Arg.
    """
    import numpy as np

    if isinstance(val, np.ndarray) or isinstance(val, (int, float, bool)):
        return mx.array(val)
    if isinstance(val, tuple):  # pragma: no branch
        return tuple(_from_numpy(r) for r in val)
    return val  # pragma: no cover


def _execute_numpy_fallback(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        cls: Arg.
        op_type: Arg.
        args: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.numpy.eager import execute_op as np_execute_op

    np_args = [_to_numpy(a) for a in args]
    np_kwargs = {k: _to_numpy(v) for k, v in kwargs.items()}
    res = np_execute_op(cls, op_type, *np_args, **np_kwargs)
    return _from_numpy(res)


def execute_op(cls: type, op_type: str, *args: object, **kwargs: object) -> object:
    """Execute execute_op.

    Args:
        cls (Any): The cls parameter for the operation.
        op_type (Any): Argument op_type.
        *args (Any): Argument *args.
        **kwargs (Any): Argument **kwargs.

    Returns:
    Any: The result.
    """
    try:
        if "dim" in kwargs and op_type not in ("TakeAlongAxis", "Take"):  # pragma: no branch
            kwargs["axis"] = kwargs.pop("dim")  # pragma: no cover

        func_registry = mlx_eager_registry.get(op_type)
        if func_registry is not None:
            return func_registry(mx, *args, **kwargs)

        return execute_generic_op(mx, op_type, *args, **kwargs)
    except (NotImplementedError, AttributeError):
        return _execute_numpy_fallback(cls, op_type, *args, **kwargs)


@mlx_eager_registry.register("TakeAlongAxis")
def _mlx_take_along_axis(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.take_along_axis(*args, **kwargs)


@mlx_eager_registry.register("Take")
def _mlx_take(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.take(*args, **kwargs)


@mlx_eager_registry.register("TensorScatterUpdate")
def _mlx_tensor_scatter_update(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = args[0], args[1], args[2]
    res = backend_module.array(tensor)
    idx = tuple(indices[..., dim] for dim in range(indices.shape[-1]))
    res[idx] = updates
    return res


@mlx_eager_registry.register("TensorScatterAdd")
def _mlx_tensor_scatter_add(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = args[0], args[1], args[2]
    import numpy as np

    res = np.array(tensor)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    np.add.at(res, idx, np.array(updates))
    return backend_module.array(res)


@mlx_eager_registry.register("TensorScatterMax")
def _mlx_tensor_scatter_max(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = args[0], args[1], args[2]
    import numpy as np

    res = np.array(tensor)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    np.maximum.at(res, idx, np.array(updates))
    return backend_module.array(res)


@mlx_eager_registry.register("TensorScatterMin")
def _mlx_tensor_scatter_min(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    tensor, indices, updates = args[0], args[1], args[2]
    import numpy as np

    res = np.array(tensor)
    idx = tuple(np.moveaxis(np.array(indices), -1, 0))
    np.minimum.at(res, idx, np.array(updates))
    return backend_module.array(res)


@mlx_eager_registry.register("ScatterNd")
def _mlx_scatter_nd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    pass


@mlx_eager_registry.register("Reshape")
def _mlx_reshape(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = kwargs.get("shape", args[1] if len(args) > 1 else kwargs.get("newshape"))
    if hasattr(shape, "data"):
        shape = shape.data
    if hasattr(shape, "tolist"):  # pragma: no branch
        shape = shape.tolist()  # pragma: no cover
    if isinstance(shape, tuple):
        shape = list(shape)
    return backend_module.reshape(args[0] if "input" not in kwargs else kwargs["input"], shape)


@mlx_eager_registry.register("Zeros")
def _mlx_zeros(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = shape.data
    dtype_val = kwargs.get("dtype", None)
    if dtype_val is None:
        return backend_module.zeros(shape)
    dtype_str = str(dtype_val).split(".")[-1]
    dtype = getattr(backend_module, dtype_str, dtype_val)
    if isinstance(shape, (int, float)):  # pragma: no branch
        shape = (int(shape),)  # pragma: no cover
    try:
        return backend_module.zeros(shape, dtype=dtype)
    except TypeError:
        return backend_module.zeros(shape)


@mlx_eager_registry.register("Ones")
def _mlx_ones(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = shape.data
    dtype_val = kwargs.get("dtype", getattr(backend_module, "float32", None))
    if dtype_val is None:
        return backend_module.ones(shape)
    dtype_str = str(dtype_val).split(".")[-1]
    dtype = getattr(backend_module, dtype_str, dtype_val)
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return backend_module.ones(shape, dtype=dtype)
    except TypeError:
        return backend_module.ones(shape)


@mlx_eager_registry.register("Full")
def _mlx_full(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else (1,))
    if hasattr(shape, "data"):
        shape = shape.data
    fill_value = kwargs.get("fill_value", args[1] if len(args) > 1 else 0)
    dtype_val = kwargs.get("dtype", getattr(backend_module, "float32", None))
    if dtype_val is None:
        return backend_module.full(shape, fill_value)
    dtype_str = str(dtype_val).split(".")[-1]
    dtype = getattr(backend_module, dtype_str, dtype_val)
    if isinstance(shape, (int, float)):
        shape = (int(shape),)
    try:
        return backend_module.full(shape, fill_value, dtype=dtype)
    except TypeError:
        return backend_module.full(shape, fill_value)


@mlx_eager_registry.register("Eye")
def _mlx_eye(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    n_arg = args[0]
    if hasattr(n_arg, "data"):  # pragma: no branch
        n_arg = n_arg.data
    return backend_module.eye(
        int(n_arg), dtype=getattr(backend_module, kwargs.get("dtype", "float32"))
    )
