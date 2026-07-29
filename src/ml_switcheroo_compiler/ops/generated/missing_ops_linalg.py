"""Missing ops for linalg."""


def eigvals(*args: object, **kwargs: object) -> object:
    """Eigvals frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Eigvals", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Eigvals", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def generalized_normal(*args: object, **kwargs: object) -> object:
    """GeneralizedNormal frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("GeneralizedNormal", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("GeneralizedNormal", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def lognormal(*args: object, **kwargs: object) -> object:
    """Lognormal frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Lognormal", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Lognormal", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def multivariate_normal(*args: object, **kwargs: object) -> object:
    """MultivariateNormal frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("MultivariateNormal", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("MultivariateNormal", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def pinv(*args: object, **kwargs: object) -> object:
    """Pinv frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Pinv", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Pinv", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def ragged_dot(*args: object, **kwargs: object) -> object:
    """RaggedDot frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("RaggedDot", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("RaggedDot", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def rayleigh(*args: object, **kwargs: object) -> object:
    """Rayleigh frontend."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Rayleigh", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Rayleigh", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


__all__ = [
    "eigvals",
    "generalized_normal",
    "lognormal",
    "multivariate_normal",
    "pinv",
    "ragged_dot",
    "rayleigh",
]
