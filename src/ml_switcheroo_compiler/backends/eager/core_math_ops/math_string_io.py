"""math_string_io module."""

from __future__ import annotations

import builtins  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager.utils import host_backend
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, global_eager_registry


@global_eager_registry.register("AsString")
def _as_string(backend_module: EagerValue, arr: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager as string operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        arr: The arr parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return str(arr)


@global_eager_registry.register("Fromfile")
def _fromfile(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fromfile operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "fromfile")(*args, **kwargs))


@global_eager_registry.register("Fromstring")
def _fromstring(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fromstring operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fromstring", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, getattr(backend_module, "fromstring")(x, **kwargs))


@global_eager_registry.register("DecodeImage")
def _np_decodeimage(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np decodeimage operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "decodeimage", getattr(backend_module, "decodeimage", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.zeros(kwargs.get("shape", (224, 224, 3)) if hasattr(kwargs, "get") else (224, 224, 3), dtype=host_backend.uint8))


@global_eager_registry.register("Fromfile")
def _np_fromfile(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fromfile operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fromfile", getattr(backend_module, "fromfile", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.fromfile(*args, **kwargs))


@global_eager_registry.register("Fromstring")
def _np_fromstring(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fromstring operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fromstring", getattr(backend_module, "fromstring", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.fromstring(*args, **kwargs))


@global_eager_registry.register("ParseTensor")
def _np_parsetensor(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np parsetensor operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "parsetensor", getattr(backend_module, "parsetensor", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.array(args[0]))


@global_eager_registry.register("StringLower")
def _np_stringlower(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np stringlower operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "stringlower", getattr(backend_module, "stringlower", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.char.lower(host_backend.asarray(args[0], dtype=str)))


@global_eager_registry.register("StringSplit")
def _np_stringsplit(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np stringsplit operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "stringsplit", getattr(backend_module, "stringsplit", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.char.split(host_backend.asarray(args[0], dtype=str)))


@global_eager_registry.register("StringToHash")
def _np_stringtohash(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np stringtohash operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "stringtohash", getattr(backend_module, "stringtohash", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.array([hash(str(x)) for x in host_backend.asarray(args[0]).flatten()]).reshape(host_backend.shape(args[0])))


@global_eager_registry.register("StringToNumber")
def _np_stringtonumber(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np stringtonumber operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "stringtonumber", getattr(backend_module, "stringtonumber", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.asarray(args[0], dtype=float))


@global_eager_registry.register("StringUpper")
def _np_stringupper(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np stringupper operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "stringupper", getattr(backend_module, "stringupper", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.char.upper(host_backend.asarray(args[0], dtype=str)))


@global_eager_registry.register("TextVectorization")
def _np_textvectorization(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np textvectorization operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "textvectorization", getattr(backend_module, "textvectorization", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("WriteFile")
def _np_writefile(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_writefile operation with atomic write and directory creation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args (filename, contents).
        **kwargs: Keyword args.

    Returns:
        int: 1 on successful write, or 0 if inputs are missing.
    """
    func = getattr(backend_module, "writefile", getattr(backend_module, "writefile", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))

    import os

    if len(args) < 2 or args[0] is None or not isinstance(args[0], (str, bytes, os.PathLike)):
        return 0

    filename = str(args[0])
    contents = args[1]

    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    tmp_path = f"{filename}.tmp.{os.getpid()}"
    try:
        if isinstance(contents, bytes):
            with open(tmp_path, "wb") as f:
                f.write(contents)
                f.flush()
                os.fsync(f.fileno())
        elif isinstance(contents, str):
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(contents)
                f.flush()
                os.fsync(f.fileno())
        elif hasattr(contents, "tobytes"):
            with open(tmp_path, "wb") as f:
                f.write(getattr(contents, "tobytes")())
                f.flush()
                os.fsync(f.fileno())
        else:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(str(contents))
                f.flush()
                os.fsync(f.fileno())

        os.replace(tmp_path, filename)
        return 1
    except OSError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
