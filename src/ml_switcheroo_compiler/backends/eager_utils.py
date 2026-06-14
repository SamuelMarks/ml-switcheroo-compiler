"""Generic utility functions for eager backend execution."""


def execute_generic_op(
    backend_module: object,
    op_type: str,
    *args: object,
    **kwargs: object,
) -> object:
    """Execute op generically.

    Args:
        backend_module (Any): The backend module (e.g., jax.numpy, mlx.core).
        op_type (str): The operation type.
        *args (object): Positional arguments for the op.
        **kwargs (object): Keyword arguments for the op.

    Returns:
        object: The result of the operation.
    """
    try:
        func = getattr(backend_module, op_type.lower())
        return func(*args, **kwargs)
    except AttributeError:
        pass

    true_divide = getattr(backend_module, "true_divide", None)
    op_map = {
        "Add": getattr(backend_module, "add", None),
        "Subtract": getattr(backend_module, "subtract", None),
        "Multiply": getattr(backend_module, "multiply", None),
        "TrueDivide": getattr(backend_module, "divide", true_divide),
        "Exp": getattr(backend_module, "exp", None),
        "Log": getattr(backend_module, "log", None),
        "Matmul": getattr(backend_module, "matmul", None),
        "Sin": getattr(backend_module, "sin", None),
        "Cos": getattr(backend_module, "cos", None),
        "Sum": getattr(backend_module, "sum", None),
        "Mean": getattr(backend_module, "mean", None),
        "Max": getattr(backend_module, "max", None),
        "Min": getattr(backend_module, "min", None),
        "Reshape": getattr(backend_module, "reshape", None),
        "Transpose": getattr(backend_module, "transpose", None),
        "Equal": getattr(backend_module, "equal", None),
        "NotEqual": getattr(backend_module, "not_equal", None),
        "Greater": getattr(backend_module, "greater", None),
        "Less": getattr(backend_module, "less", None),
        "Negative": getattr(backend_module, "negative", None),
    }

    if op_type in op_map and op_map[op_type] is not None:
        return op_map[op_type](*args, **kwargs)

    if op_type == "BroadcastTo":
        return backend_module.broadcast_to(*args, **kwargs)

    name = getattr(backend_module, "__name__", "unknown")
    msg = f"Operation '{op_type}' is not supported by backend module {name}."
    raise NotImplementedError(msg)


def generic_zeros(backend_module: object, shape: tuple[int, ...]) -> object:
    """Generic zeros function.

    Args:
        backend_module (Any): The backend module.
        shape (tuple[int, ...]): Shape of the tensor.

    Returns:
        object: A tensor of zeros.
    """
    return backend_module.zeros(shape)


def generic_array(backend_module: object, data: object) -> object:
    """Generic array creation.

    Args:
        backend_module (Any): The backend module.
        data (object): The data to convert.

    Returns:
        object: A tensor array.
    """
    try:
        return backend_module.array(data)
    except AttributeError:
        return backend_module.convert_to_tensor(data)


def generic_asarray(backend_module: object, data: object) -> object:
    """Generic asarray.

    Args:
        backend_module (Any): The backend module.
        data (object): The data to convert.

    Returns:
        object: A tensor array.
    """
    try:
        return backend_module.asarray(data)
    except AttributeError:
        return backend_module.convert_to_tensor(data)


def generic_item(backend_module: object, data: object) -> float:
    """Generic item extraction.

    Args:
        backend_module (Any): The backend module.
        data (object): The data tensor.

    Returns:
        float: The scalar value.
    """
    try:
        return float(backend_module.asarray(data).item())
    except AttributeError:
        return float(data)
