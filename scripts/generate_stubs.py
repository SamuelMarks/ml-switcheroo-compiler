"""Generate high-fidelity .pyi type stubs from ml-ecosystem-snapshots and backend schemas."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import keyword
import os

from ml_ecosystem_snapshots.stubs import (
    _format_param_list,
    _sanitize_default,
    validate_pyi_stub,
)
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.snapshot_grounding import (
    SnapshotGroundingEngine,
    _resolve_default_snapshot_dir,
)

BACKENDS_CONFIG: dict[str, str | tuple[str, ...]] = {
    "numpy": "numpy",
    "pytorch": ("torch", "torch.linalg", "torch.fft", "torch.random", "torch.nn.functional"),
    "jax": ("jax.numpy", "jax.numpy.linalg", "jax.numpy.fft", "jax.scipy.special"),
    "keras": "keras.ops",
    "mlx": ("mlx.core", "mlx.core.linalg"),
    "dask": ("dask.array", "dask.array.linalg", "dask.array.fft", "dask.array.random"),
    "cupy": ("cupy", "cupy.linalg", "cupy.fft", "cupy.random"),
    "tensorflow": ("tensorflow.math", "tensorflow.linalg", "tensorflow.random"),
    "cuda": "cuda",
    "rocm": "rocm",
    "metal": "metal",
    "awkward": "awkward",
    "dpnp": "dpnp",
    "numba": "numba",
    "sparse": "sparse",
    "pyarrow_compute": "pyarrow.compute",
    "llvm_cpp": "llvm_cpp",
    "edge": "edge",
    "edge_onnx": "onnx",
    "edge_stablehlo": "stablehlo",
    "edge_mlir": "mlir",
    "edge_wasm": "wasm",
    "edge_webgl": "webgl",
    "webgpu": "webgpu",
    "pure_python": "pure_python",
}

SAFE_TYPES: set[str] = {
    "Tensor",
    "int",
    "float",
    "bool",
    "str",
}

CORE_OPS: list[str] = [
    "abs",
    "acos",
    "acosh",
    "add",
    "all",
    "any",
    "argmax",
    "argmin",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "clip",
    "concat",
    "conv2d",
    "cos",
    "cosh",
    "cumsum",
    "divide",
    "dot",
    "equal",
    "exp",
    "floor",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "log",
    "logical_and",
    "logical_not",
    "logical_or",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "multiply",
    "negative",
    "not_equal",
    "ones",
    "pad",
    "pow",
    "reshape",
    "round",
    "rsqrt",
    "sigmoid",
    "sin",
    "sinh",
    "slice",
    "softmax",
    "split",
    "sqrt",
    "squeeze",
    "subtract",
    "sum",
    "tan",
    "tanh",
    "transpose",
    "zeros",
]

TENSOR_CLASS_STUB: list[str] = [
    "class Tensor:",
    "    shape: tuple[int, ...]",
    "    dtype: str",
    "    device: str",
    "    ndim: int",
    "    size: int",
    "    def __init__(self, data: object = ..., shape: Sequence[int] | None = ..., dtype: str | None = ...) -> None: ...",
    "    def __len__(self) -> int: ...",
    "    def __getitem__(self, item: object) -> Tensor: ...",
    "    def __add__(self, other: Tensor | float | int) -> Tensor: ...",
    "    def __sub__(self, other: Tensor | float | int) -> Tensor: ...",
    "    def __mul__(self, other: Tensor | float | int) -> Tensor: ...",
    "    def __truediv__(self, other: Tensor | float | int) -> Tensor: ...",
    "    def __neg__(self) -> Tensor: ...",
    "    def numpy(self) -> object: ...",
    "    def to(self, device: str) -> Tensor: ...",
    "    def reshape(self, shape: Sequence[int]) -> Tensor: ...",
    "    def transpose(self, axes: Sequence[int] | None = ...) -> Tensor: ...",
    "",
]

OP_SIGNATURES: dict[str, str] = {
    "abs": "(x: Tensor) -> Tensor",
    "acos": "(x: Tensor) -> Tensor",
    "acosh": "(x: Tensor) -> Tensor",
    "add": "(x1: Tensor, x2: Tensor | float | int) -> Tensor",
    "all": "(x: Tensor, axis: int | Sequence[int] | None = ..., keepdims: bool = ...) -> Tensor",
    "any": "(x: Tensor, axis: int | Sequence[int] | None = ..., keepdims: bool = ...) -> Tensor",
    "argmax": "(x: Tensor, axis: int | None = ..., keepdims: bool = ...) -> Tensor",
    "argmin": "(x: Tensor, axis: int | None = ..., keepdims: bool = ...) -> Tensor",
    "asin": "(x: Tensor) -> Tensor",
    "asinh": "(x: Tensor) -> Tensor",
    "atan": "(x: Tensor) -> Tensor",
    "atan2": "(x1: Tensor, x2: Tensor) -> Tensor",
    "atanh": "(x: Tensor) -> Tensor",
    "ceil": "(x: Tensor) -> Tensor",
    "clip": "(x: Tensor, min_val: Tensor | float, max_val: Tensor | float) -> Tensor",
    "concat": "(tensors: Sequence[Tensor], axis: int = ...) -> Tensor",
    "conv2d": "(input: Tensor, weight: Tensor, bias: Tensor | None = ..., stride: int | tuple[int, int] = ..., padding: str | int | tuple[int, int] = ...) -> Tensor",
    "cos": "(x: Tensor) -> Tensor",
    "cosh": "(x: Tensor) -> Tensor",
    "cumsum": "(x: Tensor, axis: int | None = ...) -> Tensor",
    "divide": "(x1: Tensor, x2: Tensor | float | int) -> Tensor",
    "dot": "(a: Tensor, b: Tensor) -> Tensor",
    "equal": "(x1: Tensor, x2: Tensor) -> Tensor",
    "exp": "(x: Tensor) -> Tensor",
    "floor": "(x: Tensor) -> Tensor",
    "greater": "(x1: Tensor, x2: Tensor) -> Tensor",
    "greater_equal": "(x1: Tensor, x2: Tensor) -> Tensor",
    "less": "(x1: Tensor, x2: Tensor) -> Tensor",
    "less_equal": "(x1: Tensor, x2: Tensor) -> Tensor",
    "log": "(x: Tensor) -> Tensor",
    "logical_and": "(x1: Tensor, x2: Tensor) -> Tensor",
    "logical_not": "(x: Tensor) -> Tensor",
    "logical_or": "(x1: Tensor, x2: Tensor) -> Tensor",
    "matmul": "(x1: Tensor, x2: Tensor) -> Tensor",
    "max": "(x: Tensor, axis: int | Sequence[int] | None = ..., keepdims: bool = ...) -> Tensor",
    "maximum": "(x1: Tensor, x2: Tensor) -> Tensor",
    "mean": "(x: Tensor, axis: int | Sequence[int] | None = ..., keepdims: bool = ...) -> Tensor",
    "min": "(x: Tensor, axis: int | Sequence[int] | None = ..., keepdims: bool = ...) -> Tensor",
    "minimum": "(x1: Tensor, x2: Tensor) -> Tensor",
    "multiply": "(x1: Tensor, x2: Tensor | float | int) -> Tensor",
    "negative": "(x: Tensor) -> Tensor",
    "not_equal": "(x1: Tensor, x2: Tensor) -> Tensor",
    "ones": "(shape: Sequence[int], dtype: str | None = ...) -> Tensor",
    "pad": "(x: Tensor, pad_width: Sequence[tuple[int, int]], mode: str = ...) -> Tensor",
    "pow": "(x1: Tensor, x2: Tensor | float) -> Tensor",
    "reshape": "(x: Tensor, shape: Sequence[int]) -> Tensor",
    "round": "(x: Tensor) -> Tensor",
    "rsqrt": "(x: Tensor) -> Tensor",
    "sigmoid": "(x: Tensor) -> Tensor",
    "sign": "(x: Tensor) -> Tensor",
    "sin": "(x: Tensor) -> Tensor",
    "sinh": "(x: Tensor) -> Tensor",
    "softmax": "(x: Tensor, axis: int = ...) -> Tensor",
    "sqrt": "(x: Tensor) -> Tensor",
    "square": "(x: Tensor) -> Tensor",
    "subtract": "(x1: Tensor, x2: Tensor | float | int) -> Tensor",
    "tan": "(x: Tensor) -> Tensor",
    "tanh": "(x: Tensor) -> Tensor",
    "transpose": "(x: Tensor, axes: Sequence[int] | None = ...) -> Tensor",
    "zeros": "(shape: Sequence[int], dtype: str | None = ...) -> Tensor",
}


def _validate_annot_ast(ann: ast.AST, cleaned: str) -> str:
    """Validate an annotation AST subtree against SAFE_TYPES.

    Args:
        ann (ast.AST): Parsed annotation AST node.
        cleaned (str): Cleaned annotation string.

    Returns:
        str: Validated type string or 'Tensor'.
    """
    allowed_names = SAFE_TYPES | {"Optional", "Union", "Sequence"}
    for node in ast.walk(ann):
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            return "Tensor"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value if node.value in SAFE_TYPES else "Tensor"
        if not isinstance(node, (ast.Name, ast.Constant, ast.Subscript, ast.Index, ast.Load)):
            return "Tensor"
    return cleaned


def _clean_type_annotation(raw_annot: str | None) -> str:
    """Normalize snapshot raw annotations into valid Python type stubs.

    Args:
        raw_annot (str | None): The raw annotation string.

    Returns:
        str: Normalized type annotation string.
    """
    if not raw_annot or raw_annot == "None":
        return "Tensor"

    cleaned: str = raw_annot.strip().strip("'\"").replace("array_like", "Tensor").replace("`np._NoValue`", "None")
    if any(ch in cleaned for ch in ["{", "}", "-", "(", ")", "/", "|"]) or " or " in cleaned or " and " in cleaned or (" " in cleaned and "[" not in cleaned):
        return "Tensor"

    try:
        parsed = ast.parse(f"x: {cleaned}")
        stmt = parsed.body[0]
        if isinstance(stmt, ast.AnnAssign):
            return _validate_annot_ast(stmt.annotation, cleaned)
        return "Tensor"
    except Exception:
        return "Tensor"


def format_params_via_ecosystem(
    params: list[dict[str, AttributeValue]],
    has_varargs: bool = False,
) -> str:
    """Format parameter definitions utilizing ml-ecosystem-snapshots stub formatter.

    Args:
        params (list[dict[str, AttributeValue]]): List of parameter dictionaries.
        has_varargs (bool): Whether to append generic *args if not present. Defaults to False.

    Returns:
        str: Comma-separated signature parameter string.
    """
    clean_params: list[dict[str, AttributeValue]] = []
    for p in params:
        if isinstance(p, dict) and p.get("name"):
            clean_params.append(
                {
                    "name": p["name"],
                    "kind": p.get("kind", "POSITIONAL_OR_KEYWORD"),
                    "annotation": _clean_type_annotation(str(p.get("annotation") or "Tensor")),
                    "default": p.get("default"),
                }
            )
    return str(_format_param_list(clean_params, has_varargs=has_varargs))


def _format_param_stub(param: dict[str, AttributeValue]) -> str | None:
    """Format a snapshot param dictionary into a valid Python stub parameter.

    Args:
        param (dict[str, AttributeValue]): Parameter dictionary from snapshot.

    Returns:
        Optional[str]: Formatted parameter string or None if invalid.
    """
    name: str = str(param.get("name") or "")
    if not name or name.startswith("**") or "," in name or " " in name:
        return None
    if not name.isidentifier():
        return None

    kind: str = str(param.get("kind") or "")
    raw_annot: str | None = str(param.get("annotation") or "") if param.get("annotation") else None
    typ: str = _clean_type_annotation(raw_annot)

    default_val: AttributeValue = param.get("default")
    has_default: bool = default_val is not None

    if kind == "VAR_POSITIONAL":
        return f"*{name}: {typ}"
    if kind == "VAR_KEYWORD":
        return f"**{name}: {typ}"
    if has_default:
        _sanitize_default(str(default_val))
        return f"{name}: {typ} = ..."
    return f"{name}: {typ}"


def _partition_item_params(params_data: AttributeValue) -> list[str]:
    """Partition and order formatted parameter strings from a snapshot params list.

    Args:
        params_data (AttributeValue): Raw params list from snapshot item.

    Returns:
        list[str]: Ordered parameter signature strings.
    """
    pos_args: list[str] = []
    var_pos: str | None = None
    kw_only: list[str] = []
    var_kw: str | None = None

    for p in params_data if isinstance(params_data, list) else []:
        if not isinstance(p, dict):
            continue
        formatted = _format_param_stub(p)
        if not formatted:
            continue
        kind = str(p.get("kind") or "")
        if kind == "VAR_POSITIONAL":
            var_pos = formatted
        elif kind == "VAR_KEYWORD":
            var_kw = formatted
        elif kind == "KEYWORD_ONLY":
            kw_only.append(formatted)
        else:
            pos_args.append(formatted)

    param_strs: list[str] = [p for p in pos_args if not p.endswith("= ...")] + [p for p in pos_args if p.endswith("= ...")]
    if var_pos:
        param_strs.append(var_pos)
    elif kw_only:
        param_strs.append("*")
    param_strs.extend(kw_only)
    if var_kw:
        param_strs.append(var_kw)
    return param_strs or ["*args: Tensor", "**kwargs: Tensor"]


def _format_snapshot_item_stub(item: dict[str, AttributeValue], seen_funcs: set[str]) -> str | None:
    """Format a single snapshot item dictionary into a stub function line.

    Args:
        item (dict[str, AttributeValue]): Snapshot item dictionary.
        seen_funcs (set[str]): Set of already emitted function names.

    Returns:
        str | None: Formatted stub line or None if skipped.
    """
    name: str = str(item.get("name") or "")
    item_kind = str(item.get("kind") or "")
    if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_") or name in seen_funcs or (item_kind and item_kind != "function"):
        return None

    param_strs = _partition_item_params(item.get("params", []))
    raw_ret = item.get("returns")
    ret_type: str = raw_ret if (isinstance(raw_ret, str) and raw_ret.isidentifier() and raw_ret in SAFE_TYPES) else "Tensor"
    seen_funcs.add(name)
    return f"def {name}({', '.join(param_strs)}) -> {ret_type}: ..."


def _generate_stubs_from_snapshot(data: dict[str, AttributeValue], be_name: str, out_path: str) -> int:
    """Generate high-fidelity stubs from snapshot JSON data.

    Args:
        data (dict[str, AttributeValue]): Parsed snapshot JSON data.
        be_name (str): Backend name.
        out_path (str): Destination .pyi file path.

    Returns:
        int: Number of generated function stubs.
    """
    lines: list[str] = [
        f'"""Auto-generated high-fidelity type stubs for {be_name} from ml-ecosystem-snapshots."""',
        "# ruff: noqa: E501, UP007, UP045, D100, D101, D102, D103",
        "",
        "from collections.abc import Sequence",
        "",
    ] + list(TENSOR_CLASS_STUB)

    categories = data.get("categories", {})
    seen_funcs: set[str] = {"Tensor", "Optional", "Union", "Sequence"}

    if isinstance(categories, dict):
        for items in categories.values():
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict):
                    stub_line = _format_snapshot_item_stub(item, seen_funcs)
                    if stub_line is not None:
                        lines.append(stub_line)

    lines.append("def __getattr__(name: str) -> Tensor: ...")

    stub_content: str = "\n".join(lines) + "\n"
    validate_pyi_stub(stub_content)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(stub_content)

    return len(seen_funcs) - 4


def _format_live_attr_stub(mod: AttributeValue, attr: str) -> str | None:
    """Format a live module attribute into a stub function line.

    Args:
        mod (AttributeValue): Imported module object.
        attr (str): Attribute name on module.

    Returns:
        str | None: Stub line or None if not callable.
    """
    try:
        obj = getattr(mod, attr)
        if not callable(obj):
            return None
        sig = inspect.signature(obj)
        pos_args: list[str] = []
        var_pos: str | None = None
        kw_only: list[str] = []
        var_kw: str | None = None

        for p in sig.parameters.values():
            default_str = " = ..." if p.default is not inspect.Parameter.empty else ""
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                var_pos = f"*{p.name}: Tensor"
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                var_kw = f"**{p.name}: Tensor"
            elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                kw_only.append(f"{p.name}: Tensor{default_str}")
            else:
                pos_args.append(f"{p.name}: Tensor{default_str}")

        param_strs = list(pos_args)
        if var_pos:
            param_strs.append(var_pos)
        elif kw_only:
            param_strs.append("*")
        param_strs.extend(kw_only)
        if var_kw:
            param_strs.append(var_kw)
        return f"def {attr}({', '.join(param_strs)}) -> Tensor: ..."
    except Exception:
        return f"def {attr}(*args: Tensor, **kwargs: Tensor) -> Tensor: ..."


def _generate_stubs_from_live_module(fw_name: str | tuple[str, ...], be_name: str, out_path: str) -> int:
    """Generate high-fidelity stubs by inspecting the live installed framework module.

    Args:
        fw_name (str | tuple[str, ...]): The Python module name(s) (e.g. 'torch', 'jax.numpy').
        be_name (str): Backend directory name.
        out_path (str): Destination .pyi file path.

    Returns:
        int: Number of generated function stubs.
    """
    lines: list[str] = [
        f'"""Auto-generated high-fidelity type stubs for {be_name} from live module introspection."""',
        "# ruff: noqa: E501, UP007, UP045, D100, D101, D102, D103",
        "",
        "from collections.abc import Sequence",
        "",
    ] + list(TENSOR_CLASS_STUB)

    seen_funcs: set[str] = {"Tensor", "Sequence"}
    mod_names = fw_name if isinstance(fw_name, tuple) else (fw_name,)
    for mod_name in mod_names:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        for attr in sorted(dir(mod)):
            if attr.startswith("_") or not attr.isidentifier() or keyword.iskeyword(attr) or attr in seen_funcs:
                continue
            stub_line = _format_live_attr_stub(mod, attr)
            if stub_line is not None:
                lines.append(stub_line)
                seen_funcs.add(attr)

    lines.append("def __getattr__(name: str) -> Tensor: ...")
    stub_content = "\n".join(lines) + "\n"
    if len(seen_funcs) > 2:
        validate_pyi_stub(stub_content)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(stub_content)

    return max(0, len(seen_funcs) - 2)


def _generate_fallback_backend_stubs(be_name: str, out_path: str) -> int:
    """Generate standard high-fidelity PEP 484 type stubs for a backend.

    Args:
        be_name (str): Name of backend.
        out_path (str): Destination path for stub file.

    Returns:
        int: Number of generated operation stubs.
    """
    lines: list[str] = [
        f'"""Auto-generated high-fidelity type stubs for {be_name} backend."""',
        "# ruff: noqa: E501, UP007, UP045, D100, D101, D102, D103",
        "",
        "from collections.abc import Sequence",
        "",
    ] + list(TENSOR_CLASS_STUB)

    for op in sorted(CORE_OPS):
        sig = OP_SIGNATURES.get(op, "(x: Tensor, *args: object, **kwargs: object) -> Tensor")
        lines.append(f"def {op}{sig}: ...")

    lines.append("def __getattr__(name: str) -> Tensor: ...")

    fallback_content = "\n".join(lines) + "\n"
    validate_pyi_stub(fallback_content)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(fallback_content)

    return len(CORE_OPS)


def _load_snapshot_json_file(path: str) -> dict[str, AttributeValue]:
    """Load a JSON or gzipped JSON snapshot file.

    Args:
        path (str): File path to load.

    Returns:
        dict[str, AttributeValue]: Parsed JSON dictionary.
    """
    if path.endswith(".gz"):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
            return dict(data) if isinstance(data, dict) else {}
    with open(path, encoding="utf-8") as f:
        data = json.loads(f.read())
        return dict(data) if isinstance(data, dict) else {}


def _try_generate_from_snapshot_files(
    engine: SnapshotGroundingEngine,
    snapshot_dir: str,
    be_name: str,
    fw: str | tuple[str, ...],
    stub_path: str,
) -> int:
    """Attempt stub generation from engine snapshot path or latest matching file in snapshot_dir.

    Args:
        engine (SnapshotGroundingEngine): Grounding engine instance.
        snapshot_dir (str): Snapshot directory path.
        be_name (str): Backend name.
        fw (str | tuple[str, ...]): Framework name(s).
        stub_path (str): Output stub file path.

    Returns:
        int: Number of generated stubs, or 0 if unavailable.
    """
    snapshot_file: str | None = engine.get_snapshot_path(be_name)
    if snapshot_file and os.path.exists(snapshot_file):
        try:
            return _generate_stubs_from_snapshot(_load_snapshot_json_file(snapshot_file), be_name, stub_path)
        except Exception:
            pass

    if os.path.isdir(snapshot_dir):
        snapshot_files = [f for f in os.listdir(snapshot_dir) if f.startswith(f"{fw}_v") and f.endswith((".json", ".json.gz"))]
        if snapshot_files:
            try:
                latest_path = os.path.join(snapshot_dir, sorted(snapshot_files)[-1])
                return _generate_stubs_from_snapshot(_load_snapshot_json_file(latest_path), be_name, stub_path)
            except Exception:
                pass
    return 0


def generate_stubs(
    snapshot_dir: str | None = None,
    out_base_dir: str = "src/ml_switcheroo_compiler/backends",
) -> None:
    """Generate high-fidelity .pyi stubs across all backends.

    Args:
        snapshot_dir (str | None): Directory containing snapshot JSON files.
        out_base_dir (str): Output base directory for backends.
    """
    if snapshot_dir is None:
        snapshot_dir = _resolve_default_snapshot_dir()

    if not os.path.exists(snapshot_dir):
        print(f"Snapshot directory not found: {snapshot_dir}")
        return

    engine: SnapshotGroundingEngine = SnapshotGroundingEngine(snapshot_dir)

    for be_name, fw in BACKENDS_CONFIG.items():
        be_dir: str = os.path.join(out_base_dir, be_name)
        if not os.path.exists(be_dir):
            continue
        init_file = os.path.join(be_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write(f'"""{be_name} backend module."""\n\nfrom __future__ import annotations\n')

        stub_path: str = os.path.join(be_dir, "snapshot_stubs.pyi")
        count = _try_generate_from_snapshot_files(engine, snapshot_dir, be_name, fw, stub_path)
        if count == 0:
            count = _generate_stubs_from_live_module(fw, be_name, stub_path)
        if count == 0:
            count = _generate_fallback_backend_stubs(be_name, stub_path)

        if os.path.exists(stub_path):
            print(f"Generated {stub_path} ({count} stubs).")


def main() -> None:
    """Entry point for stub generation."""
    generate_stubs()


if __name__ == "__main__":
    main()
