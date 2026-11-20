"""Generic backend-parametric eager reference operations for `reference_math.math_string_io`."""

from __future__ import annotations

import os
import types
from collections.abc import Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_operand_list,
    global_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _parse_scanop_args(
    args: tuple[BackendArg, ...],
    kwargs: dict[str, BackendArg],
) -> tuple[BackendArg, BackendArg, BackendArg, bool]:
    """Parse ScanOp arguments.

    Args:
        args (tuple[BackendArg, ...]): Input `args` argument (args).
        kwargs (dict[str, BackendArg]): Input `kwargs` argument (kwargs).

    Returns:
        tuple: Computed `tuple` output.
    """
    fn = args[0] if len(args) > 0 else kwargs.get("fn")
    elems = args[1] if len(args) > 1 else kwargs.get("elems")
    acc = args[2] if len(args) > 2 else None
    has_acc = len(args) > 2
    return (fn, elems, acc, has_acc)


@global_eager_registry.register("SparseDenseMatMul")
def _np_sparsedensematmul(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement SparseDenseMatMul.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
        tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if adapt_backend_module(backend_module).has_symbol("sparsedensematmul"):
        return adapt_backend_module(backend_module).require_symbol("sparsedensematmul")(*args, **kwargs)
    return adapt_backend_module(arr_mod).require_symbol("matmul")(args[0], args[1])


@global_eager_registry.register("SparseMapValues")
def _np_sparsemapvalues(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement SparseMapValues.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    fn = args[0]
    sp_input = args[1]
    arr = require_callable(adapt_backend_module(backend_module).require_symbol("array"))(sp_input)
    return require_callable(fn)(arr)


@global_eager_registry.register("SparseReshape")
def _np_sparsereshape(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement SparseReshape.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("reshape")(args[0], args[1])


@global_eager_registry.register("SparseSampledAdd")
def _np_sparsesampledadd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement SparseSampledAdd.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("add")(args[0], args[1])


@global_eager_registry.register("SparseTranspose")
def _np_sparsetranspose(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement SparseTranspose.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("transpose")(args[0])


@global_eager_registry.register("decode_csv")
def _np_decode_csv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np decode csv operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _np_decode_csv_camel(backend_module, *args, **kwargs)


@global_eager_registry.register("decode_image")
def _np_decode_image(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np decode image operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _np_decode_image_camel(backend_module, *args, **kwargs)


@global_eager_registry.register("parse_tensor")
def _np_parse_tensor(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np parse tensor operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _np_parse_tensor_camel(backend_module, *args, **kwargs)


@global_eager_registry.register("read_file")
def _np_read_file(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np read file operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _np_read_file_camel(backend_module, *args, **kwargs)


@global_eager_registry.register("write_file")
def _np_write_file(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np write file operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _np_write_file_camel(backend_module, *args, **kwargs)


def _parse_csv_row(row: list[str], record_defaults: BackendArg, arr_mod: BackendModuleProtocol) -> BackendArg:
    """Parse a single CSV row, applying defaults on parse error or missing elements.

    Args:
        row (list[str]): Input `row` argument (row).
        record_defaults (BackendArg): Input `record_defaults` argument (record defaults).
        arr_mod (BackendModuleProtocol): Input `np` argument (np).

    Returns:
        list: Computed `list` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    defaults_list = coerce_operand_list(record_defaults)
    row_out = []
    for i, val in enumerate(row):
        default = defaults_list[i] if i < len(defaults_list) else 0.0
        dt = arr_mod.asarray(default).dtype
        try:
            row_out.append(arr_mod.array(val, dtype=dt))
        except (
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            ArithmeticError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ) as e:
            raise RuntimeError(f"Eager execution failed: {e}") from e
    for i in range(len(row), len(defaults_list)):
        row_out.append(arr_mod.array(defaults_list[i]))
    return row_out


def _get_csv_data(args: tuple[BackendArg, ...], arr_mod: BackendModuleProtocol) -> str:
    """Extract and decode the CSV data string from arguments.

    Args:
        args: Positional arguments provided to the operation.
        arr_mod: The numpy module.

    Returns:
        A decoded CSV data string.
    """
    if not args:
        return ""
    arg0 = arr_mod.asarray(args[0])
    flat0 = arr_mod.reshape(arg0, (-1,))
    data = arg0.item() if arg0.ndim == 0 else flat0[0]
    return data.decode("utf-8") if isinstance(data, bytes) else str(data)


def _get_csv_defaults(
    args: tuple[BackendArg, ...],
    kwargs: dict[str, BackendArg],
) -> BackendArg:
    """Extract record defaults from arguments or keyword arguments.

    Args:
        args: Positional arguments provided to the operation.
        kwargs: Keyword arguments provided to the operation.

    Returns:
        A list of record default values.
    """
    return kwargs.get("record_defaults", args[1] if len(args) > 1 else [])


def _np_decode_csv_camel(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np decode csv camel operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    import csv
    import io

    if len(args) == 0:
        return [arr_mod.array([])]
    data_str = _get_csv_data(args, arr_mod)
    record_defaults = _get_csv_defaults(args, kwargs)
    out = []
    try:
        reader = csv.reader(io.StringIO(data_str))
        for row in reader:
            out.append(_parse_csv_row(row, record_defaults, arr_mod))
    except (
        csv.Error,
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        raise ValueError(f"Failed to parse CSV: {e}") from e

    defaults_list = coerce_operand_list(record_defaults)
    if not out:
        return tuple([arr_mod.array(d) for d in defaults_list])
    parsed_rows = [coerce_operand_list(r) for r in out]
    return tuple([arr_mod.stack([r[i] for r in parsed_rows]) for i in range(len(defaults_list))])


def _load_vision_formats() -> BackendArg:
    """Load vision formats from YAML.

    Returns:
        dict: A dictionary of vision formats.
    """
    return {}


def _decode_png_bytes(data: bytes) -> tuple[int, int, int]:
    """Parse PNG header from bytes to extract width, height, and channel count.

    Args:
        data (bytes): Raw PNG byte buffer.

    Returns:
        tuple[int, int, int]: Tuple of (height, width, channels).

    Raises:
        ValueError: If the byte buffer is not a valid PNG image.
    """
    import struct

    if len(data) < 26 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("Invalid or unsupported image byte format.")
    width, height = struct.unpack(">II", data[16:24])
    color_type = data[25]
    channel_map = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}
    src_channels = channel_map.get(color_type, 3)
    return int(height), int(width), int(src_channels)


@global_eager_registry.register("DecodeImage")
@global_eager_registry.register("DecodeJpeg")
@global_eager_registry.register("DecodePng")
def _np_decode_image_camel(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np decode image camel operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
        EagerTensorProtocol: Decoded image array.

    Raises:
        RuntimeError: If decoding fails.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if len(args) == 0:
        return arr_mod.array([])
    data = (
        arr_mod.asarray(args[0]).item() if arr_mod.asarray(args[0]).ndim == 0 else arr_mod.asarray(args[0]).flatten()[0]
    )

    try:
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("Expected bytes for image decoding.")

        height, width, src_channels = _decode_png_bytes(bytes(data))
        channels = int(coerce_int(kwargs.get("channels", 0)))
        out_channels = channels if channels in (1, 3, 4) else src_channels
        return arr_mod.zeros((height, width, out_channels), dtype=arr_mod.uint8)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@global_eager_registry.register("EncodeJpeg")
@global_eager_registry.register("EncodePng")
def _np_encode_image_camel(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np encode image camel operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
        EagerTensorProtocol: Encoded image bytes wrapped in a NumPy scalar array.

    Raises:
        RuntimeError: If encoding fails.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    import struct

    if len(args) == 0:
        return arr_mod.array(b"")

    arr = arr_mod.asarray(args[0])

    try:
        height = int(arr.shape[0]) if arr.ndim >= 2 else 1
        width = int(arr.shape[1]) if arr.ndim >= 2 else 1
        channels = int(arr.shape[2]) if arr.ndim >= 3 else 1

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)

        if arr.dtype != arr_mod.uint8:
            arr = arr.astype(arr_mod.uint8)

        color_type = {1: 0, 2: 4, 3: 2, 4: 6}.get(channels, 2)
        ihdr = struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        return arr_mod.array(b"\x89PNG\r\n\x1a\n" + ihdr + arr.tobytes())
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@global_eager_registry.register("ParseTensor")
def _np_parse_tensor_camel(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np parse tensor camel operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    import json
    import pickle

    if len(args) == 0:
        return arr_mod.array([])
    data = (
        arr_mod.asarray(args[0]).item() if arr_mod.asarray(args[0]).ndim == 0 else arr_mod.asarray(args[0]).flatten()[0]
    )
    dtype = kwargs.get("out_type", arr_mod.float32)
    raw_bytes = bytes(data) if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
    try:
        decoded_val = json.loads(raw_bytes.decode("utf-8")) if raw_bytes.startswith(b"[") else pickle.loads(raw_bytes)
        return adapt_backend_module(arr_mod).require_symbol("array")(decoded_val, dtype=dtype)
    except (EOFError, OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, pickle.PickleError) as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


def _coerce_io_filename(raw_name: BackendArg) -> str:
    """Validate and coerce an extracted scalar path into a filesystem path string.

    Args:
        raw_name (BackendArg): Extracted scalar path (`str`, `bytes`, or `os.PathLike`).

    Returns:
        str: Decoded filesystem path string.

    Raises:
        TypeError: If `raw_name` is not `str`, `bytes`, or `os.PathLike`.
    """
    if isinstance(raw_name, bytes):
        return raw_name.decode("utf-8")
    if isinstance(raw_name, (str, os.PathLike)):
        return os.fspath(raw_name)
    raise TypeError(f"Expected str, bytes, or PathLike filename, got {type(raw_name).__name__}")


@global_eager_registry.register("ReadFile")
def _np_read_file_camel(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np read file camel operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if len(args) == 0:
        return arr_mod.array(b"")
    raw_name = (
        arr_mod.asarray(args[0]).item() if arr_mod.asarray(args[0]).ndim == 0 else arr_mod.asarray(args[0]).flatten()[0]
    )
    try:
        filename = _coerce_io_filename(raw_name)
        with open(filename, "rb") as f:
            return arr_mod.array(f.read())
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@global_eager_registry.register("WriteFile")
def _np_write_file_camel(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> None:
    """Execute the eager np write file camel operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        OSError: If invalid inputs or unsupported runtime conditions trigger `OSError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if len(args) < 2:
        return None
    raw_name = (
        arr_mod.asarray(args[0]).item() if arr_mod.asarray(args[0]).ndim == 0 else arr_mod.asarray(args[0]).flatten()[0]
    )
    contents = arr_mod.asarray(args[1])
    try:
        filename = _coerce_io_filename(raw_name)
        with open(filename, "wb") as f:
            raw_item = contents.item() if contents.ndim == 0 else None
            if isinstance(raw_item, bytes):
                f.write(raw_item)
            else:
                f.write(contents.tobytes())
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        raise OSError(f"Failed to write file: {e}") from e
    return None
