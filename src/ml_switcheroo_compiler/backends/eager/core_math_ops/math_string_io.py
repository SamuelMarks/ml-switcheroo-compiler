"""Implementation of math string io operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_string_io`)."""

from __future__ import annotations

import builtins  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_tensor, global_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("AsString")
def _as_string(backend_module: BackendModuleProtocol, arr: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _fromfile(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fromfile operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return as_tensor(getattr(backend_module, "fromfile")(*args, **kwargs))


@global_eager_registry.register("Fromstring")
def _fromstring(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    x = args[0]
    return as_tensor(getattr(backend_module, "fromstring")(x, **kwargs))


@global_eager_registry.register("DecodeImage")
def _np_decodeimage(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.zeros(kwargs.get("shape", (224, 224, 3)) if hasattr(kwargs, "get") else (224, 224, 3), dtype=mod.uint8)


@global_eager_registry.register("Fromfile")
def _np_fromfile(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).fromfile(*args, **kwargs)


@global_eager_registry.register("Fromstring")
def _np_fromstring(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).fromstring(*args, **kwargs)


@global_eager_registry.register("ParseTensor")
def _np_parsetensor(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).array(args[0])


@global_eager_registry.register("StringLower")
def _np_stringlower(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.char.lower(mod.asarray(args[0], dtype=str))


@global_eager_registry.register("StringSplit")
def _np_stringsplit(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.char.split(mod.asarray(args[0], dtype=str))


@global_eager_registry.register("StringToHash")
def _np_stringtohash(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.array([hash(str(x)) for x in mod.asarray(args[0]).flatten()]).reshape(tuple(int(d) for d in mod.shape(args[0])))


@global_eager_registry.register("StringToNumber")
def _np_stringtonumber(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).asarray(args[0], dtype=float)


@global_eager_registry.register("StringUpper")
def _np_stringupper(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.char.upper(mod.asarray(args[0], dtype=str))


@global_eager_registry.register("TextVectorization")
def _np_textvectorization(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("WriteFile")
def _np_writefile(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
        return as_tensor(func(*args, **kwargs))

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
