"""Missing ops for math."""


def double_sided_maxwell(*args: object, **kwargs: object) -> object:
    """DoubleSidedMaxwell frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("DoubleSidedMaxwell", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("DoubleSidedMaxwell", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def hamming(*args: object, **kwargs: object) -> object:
    """Hamming frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Hamming", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Hamming", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def householder_product(*args: object, **kwargs: object) -> object:
    """HouseholderProduct frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("HouseholderProduct", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("HouseholderProduct", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def loggamma(*args: object, **kwargs: object) -> object:
    """Loggamma frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Loggamma", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Loggamma", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def logistic(*args: object, **kwargs: object) -> object:
    """Logistic frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Logistic", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Logistic", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def matrix_power(*args: object, **kwargs: object) -> object:
    """MatrixPower frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("MatrixPower", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("MatrixPower", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def maxwell(*args: object, **kwargs: object) -> object:
    """Maxwell frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Maxwell", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Maxwell", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def polyadd(*args: object, **kwargs: object) -> object:
    """Polyadd frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Polyadd", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Polyadd", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def polydiv(*args: object, **kwargs: object) -> object:
    """Polydiv frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Polydiv", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Polydiv", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def polymul(*args: object, **kwargs: object) -> object:
    """Polymul frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Polymul", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Polymul", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def polysub(*args: object, **kwargs: object) -> object:
    """Polysub frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Polysub", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("Polysub", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


def weibull_min(*args: object, **kwargs: object) -> object:
    """WeibullMin frontend.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        object: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("WeibullMin", *args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node("WeibullMin", list(args), kwargs, getattr(args[0], "shape", ()) if args else (), getattr(args[0], "dtype", "float32") if args else "float32")


__all__ = [
    "double_sided_maxwell",
    "hamming",
    "householder_product",
    "loggamma",
    "logistic",
    "matrix_power",
    "maxwell",
    "polyadd",
    "polydiv",
    "polymul",
    "polysub",
    "weibull_min",
]
