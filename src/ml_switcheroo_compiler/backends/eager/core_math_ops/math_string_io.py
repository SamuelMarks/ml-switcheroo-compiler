"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_string_io`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_bytes,
    global_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("AsString")
def _as_string(
    backend_module: BackendModuleProtocol | types.ModuleType | None, arr: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager as string operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        arr: Input `arr` argument (arr).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return str(arr)


@global_eager_registry.register("Fromfile")
def _fromfile(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fromfile operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("fromfile")(*args, **kwargs)


@global_eager_registry.register("Fromstring")
def _fromstring(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fromstring operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("fromstring", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return adapt_backend_module(backend_module).require_symbol("fromstring")(x, **kwargs)


@global_eager_registry.register("DecodeImage")
def _np_decodeimage(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np decodeimage operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "decodeimage", adapt_backend_module(backend_module).get_symbol("decodeimage", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.zeros(
        kwargs.get("shape", (224, 224, 3)),
        dtype=mod.uint8,
    )


@global_eager_registry.register("Fromfile")
def _np_fromfile(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fromfile operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "fromfile", adapt_backend_module(backend_module).get_symbol("fromfile", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).fromfile(*args, **kwargs)


@global_eager_registry.register("Fromstring")
def _np_fromstring(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fromstring operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "fromstring", adapt_backend_module(backend_module).get_symbol("fromstring", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).fromstring(*args, **kwargs)


@global_eager_registry.register("DecodeCsv")
def _generic_decode_csv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Decode delimited CSV strings into column arrays on the active backend.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend execution module.
        *args (BackendArg): Positional CSV string tensor and optional defaults.
        **kwargs (BackendArg): Delimiter and ``record_defaults`` options.

    Returns:
        BackendArg: Tuple of decoded column arrays.
    """
    import csv
    import io

    mod = _resolve_eager_mod(backend_module)
    first_arg = args[0] if args else ""
    raw = "\n".join(str(x) for x in first_arg) if isinstance(first_arg, (list, tuple)) else str(first_arg)
    delim_raw = kwargs.get("field_delim", ",")
    delim = delim_raw if isinstance(delim_raw, str) and len(delim_raw) == 1 else ","
    defaults_raw = kwargs.get("record_defaults", args[1] if len(args) > 1 else ())
    defaults = list(defaults_raw) if isinstance(defaults_raw, (list, tuple)) else []
    rows = list(csv.reader(io.StringIO(raw), delimiter=delim))
    if not rows or not defaults:
        return tuple(mod.asarray(d) for d in defaults)
    cols: list[BackendArg] = []
    for col_idx, default_val in enumerate(defaults):
        col_vals = [float(r[col_idx]) if col_idx < len(r) and r[col_idx] != "" else default_val for r in rows]
        cols.append(mod.asarray(col_vals))
    return tuple(cols)


@global_eager_registry.register("SerializeTensor")
def _generic_serialize_tensor(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Serialize a tensor operand into a UTF-8 JSON byte payload on the active backend.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend execution module.
        *args (BackendArg): Positional tensor operand to serialize.
        **kwargs (BackendArg): Optional operation attributes.

    Returns:
        BackendArg: UTF-8 encoded JSON bytes representing the tensor values.
    """
    import json

    del backend_module, kwargs
    if not args:
        return b"[]"
    target = args[0]
    tolist_fn = adapt_backend_module(target).get_symbol("tolist", None)
    payload = tolist_fn() if callable(tolist_fn) else target
    return json.dumps(payload).encode("utf-8")


@global_eager_registry.register("ParseTensor")
def _np_parsetensor(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np parsetensor operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "parsetensor", adapt_backend_module(backend_module).get_symbol("parsetensor", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).array(args[0])


@global_eager_registry.register("StringLower")
def _np_stringlower(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np stringlower operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "stringlower", adapt_backend_module(backend_module).get_symbol("stringlower", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.char.lower(mod.asarray(args[0], dtype=str))


@global_eager_registry.register("StringSplit")
def _np_stringsplit(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np stringsplit operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "stringsplit", adapt_backend_module(backend_module).get_symbol("stringsplit", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.char.split(mod.asarray(args[0], dtype=str))


@global_eager_registry.register("StringToHash")
def _np_stringtohash(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np stringtohash operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "stringtohash", adapt_backend_module(backend_module).get_symbol("stringtohash", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_shape

    target_shape = coerce_shape(require_callable(mod.shape)(args[0]))
    return mod.array([hash(str(x)) for x in mod.asarray(args[0]).flatten()]).reshape(target_shape)


@global_eager_registry.register("StringToNumber")
def _np_stringtonumber(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np stringtonumber operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "stringtonumber", adapt_backend_module(backend_module).get_symbol("stringtonumber", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).asarray(args[0], dtype=float)


@global_eager_registry.register("StringUpper")
def _np_stringupper(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np stringupper operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "stringupper", adapt_backend_module(backend_module).get_symbol("stringupper", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.char.upper(mod.asarray(args[0], dtype=str))


@global_eager_registry.register("TextVectorization")
def _np_textvectorization(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np textvectorization operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "textvectorization", adapt_backend_module(backend_module).get_symbol("textvectorization", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("WriteFile")
def _np_writefile(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate _np_writefile operation with atomic write and directory creation.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args (filename, contents).
        **kwargs: Keyword args.

    Returns:
        int: 1 on successful write, or 0 if inputs are missing.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "writefile", adapt_backend_module(backend_module).get_symbol("writefile", None)
    )
    if func is not None:
        return func(*args, **kwargs)

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
        elif adapt_backend_module(contents).has_symbol("tobytes"):
            with open(tmp_path, "wb") as f:
                f.write(coerce_bytes(adapt_backend_module(contents).require_symbol("tobytes")()))
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
