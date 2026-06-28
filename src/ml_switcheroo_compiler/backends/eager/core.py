# ruff: noqa: E402, F403
"""Core eager operations."""

"""Core utilities."""


import warnings


from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def execute_generic_op(  # noqa: C901, PLR0911
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
        # Check if the module has a random submodule
        try:
            import jax.random
            import jax.scipy.special
            import scipy.special

            # For random distributions, try to find them in jax.random or numpy.random
            op_lower = op_type.lower()
            if hasattr(jax.random, op_lower):
                return getattr(jax.random, op_lower)(*args, **kwargs)  # pragma: no cover

            # Map specific things to scipy.special
            if op_type == "Erfinv":
                return scipy.special.erfinv(*args, **kwargs)  # pragma: no cover

            # Random functions might be camelcased differently
            op_camel_to_snake = "".join(
                ["_" + c.lower() if c.isupper() else c for c in op_type]
            ).lstrip("_")
            if hasattr(jax.random, op_camel_to_snake):
                return getattr(jax.random, op_camel_to_snake)(*args, **kwargs)  # pragma: no cover
        # pragma: no cover
        except ImportError:  # pragma: no cover
            pass  # pragma: no cover

        # Try linalg specifically
        try:
            if op_type == "CustomLinearSolve":
                if hasattr(backend_module, "linalg") and hasattr(
                    backend_module.linalg, "solve"
                ):  # pragma: no cover
                    return backend_module.linalg.solve(*args, **kwargs)  # pragma: no cover
                import jax.scipy.linalg  # pragma: no cover

                # pragma: no cover
                return jax.scipy.linalg.solve(*args, **kwargs)  # pragma: no cover
        except ImportError:  # pragma: no cover
            pass  # pragma: no cover

    name = getattr(backend_module, "__name__", "unknown")
    msg = f"Operation '{op_type}' is not supported by backend module {name}."
    raise NotImplementedError(msg)
