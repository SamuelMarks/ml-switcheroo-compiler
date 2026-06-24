# ruff: noqa: E402, F403
"""Core eager operations."""

"""Core utilities."""


import warnings


from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


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
    warnings.warn(
        "Python-level eager execution bypasses and _EAGER_OP_MAP fallbacks are deprecated. "
        "Frontends should only trace to LogicalNode. "
        "Graph execution should be isolated to eval() phase.",
        DeprecationWarning,
        stacklevel=2,
    )

    func_registry = global_eager_registry.get(op_type)
    if func_registry is not None:
        return func_registry(backend_module, *args, **kwargs)

    try:
        func = getattr(backend_module, op_type.lower())
        if op_type not in (
            "Sort",
            "ArgSort",
            "Allclose",
            "Reshape",
            "BroadcastTo",
        ):  # pragma: no branch
            return func(*args, **kwargs)
    except AttributeError:
        pass

    name = getattr(backend_module, "__name__", "unknown")
    msg = f"Operation '{op_type}' is not supported by backend module {name}."
    raise NotImplementedError(msg)
