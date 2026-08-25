# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_string_io module."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("AsString")
def _as_string(backend_module: object, arr: object, **kwargs: object) -> object:
    """Evaluate _as_string operation.

    Args:
        backend_module (object): The backend_module parameter.
        arr (object): The arr parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return str(arr)


@global_eager_registry.register("Fromfile")
def _fromfile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _fromfile operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.fromfile(*args, **kwargs)


@global_eager_registry.register("Fromstring")
def _fromstring(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _fromstring operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "fromstring", None)
    if func:
        return func(*args, **kwargs)
    x: object = args[0]
    return backend_module.fromstring(x, **kwargs)


@global_eager_registry.register("DecodeImage")
def _np_decodeimage(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_decodeimage operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "decodeimage", getattr(backend_module, "decodeimage", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.zeros(kwargs.get("shape", (224, 224, 3)), dtype=np.uint8)


@global_eager_registry.register("Fromfile")
def _np_fromfile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_fromfile operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "fromfile", getattr(backend_module, "fromfile", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.fromfile(*args, **kwargs)


@global_eager_registry.register("Fromstring")
def _np_fromstring(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_fromstring operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "fromstring", getattr(backend_module, "fromstring", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.fromstring(*args, **kwargs)


@global_eager_registry.register("ParseTensor")
def _np_parsetensor(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_parsetensor operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "parsetensor", getattr(backend_module, "parsetensor", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.array(args[0])


@global_eager_registry.register("StringLower")
def _np_stringlower(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_stringlower operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "stringlower", getattr(backend_module, "stringlower", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.char.lower(np.asarray(args[0], dtype=str))


@global_eager_registry.register("StringSplit")
def _np_stringsplit(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_stringsplit operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "stringsplit", getattr(backend_module, "stringsplit", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.char.split(np.asarray(args[0], dtype=str))


@global_eager_registry.register("StringToHash")
def _np_stringtohash(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_stringtohash operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "stringtohash", getattr(backend_module, "stringtohash", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.array([hash(str(x)) for x in np.asarray(args[0]).flatten()]).reshape(np.shape(args[0]))


@global_eager_registry.register("StringToNumber")
def _np_stringtonumber(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_stringtonumber operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "stringtonumber", getattr(backend_module, "stringtonumber", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.asarray(args[0], dtype=float)


@global_eager_registry.register("StringUpper")
def _np_stringupper(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_stringupper operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "stringupper", getattr(backend_module, "stringupper", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.char.upper(np.asarray(args[0], dtype=str))


@global_eager_registry.register("TextVectorization")
def _np_textvectorization(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_textvectorization operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "textvectorization", getattr(backend_module, "textvectorization", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return args[0]


@global_eager_registry.register("WriteFile")
def _np_writefile(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_writefile operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "writefile", getattr(backend_module, "writefile", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return None
