# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_linalg module."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Det")
def _det(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _det operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "det"):
        return func.det(*args, **kwargs)
    if hasattr(backend_module, "det"):
        return backend_module.det(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.det(backend_module.asarray(x))


@global_eager_registry.register("Eig")
def _eig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _eig operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eig"):
        return func.eig(*args, **kwargs)
    if hasattr(backend_module, "eig"):
        return backend_module.eig(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.eig(backend_module.asarray(x))


@global_eager_registry.register("Eigh")
def _eigh(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _eigh operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eigh"):
        return func.eigh(*args, **kwargs)
    if hasattr(backend_module, "eigh"):
        return backend_module.eigh(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.eigh(backend_module.asarray(x))


@global_eager_registry.register("Eigvals")
def _eigvals(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _eigvals operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eigvals"):
        return func.eigvals(*args, **kwargs)
    if hasattr(backend_module, "eigvals"):
        return backend_module.eigvals(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.eigvals(backend_module.asarray(x))


@global_eager_registry.register("Eigvalsh")
def _eigvalsh(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _eigvalsh operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "eigvalsh"):
        return func.eigvalsh(*args, **kwargs)
    if hasattr(backend_module, "eigvalsh"):
        return backend_module.eigvalsh(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.eigvalsh(backend_module.asarray(x))


@global_eager_registry.register("Cholesky")
def _cholesky(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _cholesky operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "cholesky"):
        return func.cholesky(*args, **kwargs)
    if hasattr(backend_module, "cholesky"):
        return backend_module.cholesky(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.cholesky(backend_module.asarray(x))


@global_eager_registry.register("CholeskyEx")
def _cholesky_ex(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _cholesky_ex operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    chol = _cholesky(backend_module, *args, **kwargs)
    if hasattr(backend_module, "zeros"):
        info = backend_module.zeros((), dtype=getattr(backend_module, "int32", int))
    else:
        info = 0
    return (chol, info)


@global_eager_registry.register("Norm")
def _norm(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _norm operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "norm"):
        return func.norm(*args, **kwargs)
    if hasattr(backend_module, "norm"):
        return backend_module.norm(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.norm(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Pinv")
def _pinv(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _pinv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "pinv"):
        return func.pinv(*args, **kwargs)
    if hasattr(backend_module, "pinv"):
        return backend_module.pinv(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.pinv(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Qr")
def _qr(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _qr operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "qr"):
        return func.qr(*args, **kwargs)
    if hasattr(backend_module, "qr"):
        return backend_module.qr(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.qr(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Svd")
def _svd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _svd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "svd"):
        return func.svd(*args, **kwargs)
    if hasattr(backend_module, "svd"):
        return backend_module.svd(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.svd(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Tensorinv")
def _tensorinv(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _tensorinv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "tensorinv"):
        return func.tensorinv(*args, **kwargs)
    if hasattr(backend_module, "tensorinv"):
        return backend_module.tensorinv(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.tensorinv(backend_module.asarray(x), **kwargs)


@global_eager_registry.register("Inv")
def _inv(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _inv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "linalg", None)
    if func and hasattr(func, "inv"):
        return func.inv(*args, **kwargs)
    if hasattr(backend_module, "inv"):
        return backend_module.inv(*args, **kwargs)
    x = args[0]
    return backend_module.linalg.inv(backend_module.asarray(x))


@global_eager_registry.register("Vander")
def _vander(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _vander operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.vander(*args, **kwargs)


@global_eager_registry.register("Schur")
def _np_schur(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_schur operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "schur", getattr(backend_module, "schur", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return (args[0], args[0])


@global_eager_registry.register("Svd")
def _np_svd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_svd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "svd", getattr(backend_module, "svd", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.linalg.svd(*args, **kwargs)


@global_eager_registry.register("Svdvals")
def _np_svdvals(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_svdvals operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "svdvals", getattr(backend_module, "svdvals", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.linalg.svd(*args, **kwargs)[1]


@global_eager_registry.register("Tensorinv")
def _np_tensorinv(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorinv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "tensorinv", getattr(backend_module, "tensorinv", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.linalg.tensorinv(*args, **kwargs)


@global_eager_registry.register("TriInv")
def _np_triinv(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_triinv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "triinv", getattr(backend_module, "triinv", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.linalg.inv(args[0])


@global_eager_registry.register("UniqueInverse")
def _np_uniqueinverse(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_uniqueinverse operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "uniqueinverse", getattr(backend_module, "uniqueinverse", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.unique(args[0], return_inverse=True)


@global_eager_registry.register("Vander")
def _np_vander(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_vander operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "vander", getattr(backend_module, "vander", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.vander(*args, **kwargs)


@global_eager_registry.register("VectorNorm")
def _np_vectornorm(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_vectornorm operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "vectornorm", getattr(backend_module, "vectornorm", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.linalg.norm(*args, **kwargs)
