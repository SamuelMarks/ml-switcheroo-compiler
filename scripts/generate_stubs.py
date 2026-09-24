"""Generate high-fidelity .pyi type stubs from ml-framework-snapshots and backend schemas."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import keyword
import os
import shutil

BACKENDS_CONFIG: dict[str, str] = {
    "numpy": "numpy",
    "pytorch": "torch",
    "jax": "jax.numpy",
    "keras": "keras.ops",
    "mlx": "mlx.core",
    "dask": "dask.array",
    "cupy": "cupy",
    "tensorflow": "tensorflow.math",
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
}

SAFE_TYPES: set[str] = {
    "Tensor",
    "int",
    "float",
    "bool",
    "str",
    "None",
    "tuple",
    "dict",
    "list",
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


def _clean_type_annotation(raw_annot: str | None) -> str:
    """Normalize snapshot raw annotations into valid Python type stubs.

    Args:
        raw_annot (Optional[str]): The raw annotation string.

    Returns:
        str: Normalized type annotation string.
    """
    if not raw_annot or raw_annot == "None":
        return "Tensor"

    cleaned: str = raw_annot.strip().replace("array_like", "Tensor")
    cleaned = cleaned.replace("`np._NoValue`", "None")

    if any(ch in cleaned for ch in ["{", "}", "-", "(", ")", "/", "|"]):
        return "Tensor"

    try:
        parsed = ast.parse(f"x: {cleaned}")
        ann = parsed.body[0].annotation  # type: ignore[attr-defined]
        for node in ast.walk(ann):
            if isinstance(node, ast.Name):
                if node.id not in SAFE_TYPES and node.id not in {"Optional", "Union", "Sequence", "tuple"}:
                    return "Tensor"
        return cleaned
    except Exception:
        return "Tensor"


def _format_param_stub(param: dict[str, object]) -> str | None:
    """Format a snapshot param dictionary into a valid Python stub parameter.

    Args:
        param (dict[str, object]): Parameter dictionary from snapshot.

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

    default_val: object | None = param.get("default")
    has_default: bool = default_val is not None

    if kind == "VAR_POSITIONAL":
        return f"*{name}: {typ}"
    if kind == "VAR_KEYWORD":
        return f"**{name}: {typ}"
    if has_default:
        return f"{name}: {typ} = ..."
    return f"{name}: {typ}"


def _generate_stubs_from_snapshot(data: dict[str, object], be_name: str, out_path: str) -> int:
    """Generate high-fidelity stubs from snapshot JSON data.

    Args:
        data (dict[str, object]): Parsed snapshot JSON data.
        be_name (str): Backend name.
        out_path (str): Destination .pyi file path.

    Returns:
        int: Number of generated function stubs.
    """
    lines: list[str] = [
        f'"""Auto-generated high-fidelity type stubs for {be_name} from ml-framework-snapshots."""',
        "# ruff: noqa: E501",
        "",
        "from collections.abc import Sequence",
        "from typing import Optional, Union",
        "",
        "class Tensor:",
        "    shape: tuple[int, ...]",
        "    dtype: str",
        "    def __init__(self, *args: object, **kwargs: object) -> None: ...",
        "",
    ]

    categories: dict[str, object] = data.get("categories", {})  # type: ignore[assignment]
    seen_funcs: set[str] = {"Tensor", "Optional", "Union", "Sequence"}

    for _, items in categories.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            name: str = str(item.get("name") or "")
            if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_") or name in seen_funcs:
                continue

            item_kind = str(item.get("kind") or "")
            if item_kind and item_kind != "function":
                continue

            params_data = item.get("params", [])
            pos_args: list[str] = []
            var_pos: str | None = None
            kw_only: list[str] = []
            var_kw: str | None = None

            if isinstance(params_data, list):
                for p in params_data:
                    if not isinstance(p, dict):
                        continue
                    kind = str(p.get("kind") or "")
                    formatted = _format_param_stub(p)
                    if not formatted:
                        continue
                    if kind == "VAR_POSITIONAL":
                        var_pos = formatted
                    elif kind == "VAR_KEYWORD":
                        var_kw = formatted
                    elif kind == "KEYWORD_ONLY":
                        kw_only.append(formatted)
                    else:
                        pos_args.append(formatted)

            pos_no_default: list[str] = [p for p in pos_args if not p.endswith("= ...")]
            pos_with_default: list[str] = [p for p in pos_args if p.endswith("= ...")]
            param_strs: list[str] = pos_no_default + pos_with_default
            if var_pos:
                param_strs.append(var_pos)
            elif kw_only:
                param_strs.append("*")
            param_strs.extend(kw_only)
            if var_kw:
                param_strs.append(var_kw)

            if not param_strs:
                param_strs = ["*args: Tensor", "**kwargs: Tensor"]

            ret_type: str = "Tensor"
            raw_ret = item.get("returns")
            if isinstance(raw_ret, str) and raw_ret.isidentifier() and raw_ret in SAFE_TYPES:
                ret_type = raw_ret

            params_joined = ", ".join(param_strs)
            lines.append(f"def {name}({params_joined}) -> {ret_type}: ...")
            seen_funcs.add(name)

    lines.append("def __getattr__(name: str) -> Tensor: ...")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return len(seen_funcs) - 4


def _generate_stubs_from_live_module(fw_name: str, be_name: str, out_path: str) -> int:
    """Generate high-fidelity stubs by inspecting the live installed framework module.

    Args:
        fw_name (str): The Python module name (e.g. 'torch', 'jax.numpy').
        be_name (str): Backend directory name.
        out_path (str): Destination .pyi file path.

    Returns:
        int: Number of generated function stubs.
    """
    lines: list[str] = [
        f'"""Auto-generated high-fidelity type stubs for {be_name} from live module introspection."""',
        "# ruff: noqa: E501",
        "",
        "from collections.abc import Sequence",
        "from typing import Optional, Union",
        "",
        "class Tensor:",
        "    shape: tuple[int, ...]",
        "    dtype: str",
        "    def __init__(self, *args: object, **kwargs: object) -> None: ...",
        "",
    ]

    try:
        mod = importlib.import_module(fw_name)
    except Exception:
        return _generate_fallback_backend_stubs(be_name, out_path)

    seen_funcs: set[str] = {"Tensor"}
    for attr in sorted(dir(mod)):
        if attr.startswith("_") or not attr.isidentifier() or keyword.iskeyword(attr) or attr in seen_funcs:
            continue
        try:
            obj = getattr(mod, attr)
            if not callable(obj):
                continue
            sig = inspect.signature(obj)
            pos_args = []
            var_pos = None
            kw_only = []
            var_kw = None

            for p in sig.parameters.values():
                p_name = p.name
                default_str = " = ..." if p.default is not inspect.Parameter.empty else ""
                if p.kind == inspect.Parameter.VAR_POSITIONAL:
                    var_pos = f"*{p_name}: Tensor"
                elif p.kind == inspect.Parameter.VAR_KEYWORD:
                    var_kw = f"**{p_name}: Tensor"
                elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                    kw_only.append(f"{p_name}: Tensor{default_str}")
                else:
                    pos_args.append(f"{p_name}: Tensor{default_str}")

            param_strs = list(pos_args)
            if var_pos:
                param_strs.append(var_pos)
            elif kw_only:
                param_strs.append("*")
            param_strs.extend(kw_only)
            if var_kw:
                param_strs.append(var_kw)

            joined = ", ".join(param_strs)
            lines.append(f"def {attr}({joined}) -> Tensor: ...")
            seen_funcs.add(attr)
        except Exception:
            lines.append(f"def {attr}(*args: Tensor, **kwargs: Tensor) -> Tensor: ...")
            seen_funcs.add(attr)

    lines.append("def __getattr__(name: str) -> Tensor: ...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return len(seen_funcs) - 1


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
        "# ruff: noqa: E501",
        "",
        "from collections.abc import Sequence",
        "from typing import Optional, Union",
        "",
        "class Tensor:",
        "    shape: tuple[int, ...]",
        "    dtype: str",
        "    def __init__(self, *args: object, **kwargs: object) -> None: ...",
        "",
    ]

    for op in sorted(CORE_OPS):
        lines.append(f"def {op}(*args: Tensor, **kwargs: Tensor) -> Tensor: ...")

    lines.append("def __getattr__(name: str) -> Tensor: ...")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return len(CORE_OPS)


def generate_stubs(
    snapshot_dir: str | None = None,
    out_base_dir: str = "src/ml_switcheroo_compiler/backends",
) -> None:
    """Generate high-fidelity .pyi stubs across all backends.

    Args:
        snapshot_dir (Optional[str]): Directory containing snapshot JSON files.
        out_base_dir (str): Output base directory for backends.
    """
    if snapshot_dir is None:
        snapshot_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "ml-framework-snapshots",
                "src",
                "ml_framework_snapshots",
                "snapshots",
            )
        )

    for be_name, fw in BACKENDS_CONFIG.items():
        be_dir: str = os.path.join(out_base_dir, be_name)
        os.makedirs(be_dir, exist_ok=True)
        init_file = os.path.join(be_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as f:
                f.write(f'"""{be_name} backend module."""\n\nfrom __future__ import annotations\n')

        stub_path: str = os.path.join(be_dir, "snapshot_stubs.pyi")
        alt_stub_path: str = os.path.join(be_dir, "stub.pyi")
        count: int = 0

        snapshot_files: list[str] = []
        if os.path.isdir(snapshot_dir):
            snapshot_files = [f for f in os.listdir(snapshot_dir) if f.startswith(f"{fw}_v") and f.endswith(".json")]

        if snapshot_files:
            latest_file = sorted(snapshot_files)[-1]
            try:
                with open(os.path.join(snapshot_dir, latest_file), encoding="utf-8") as f:
                    data = json.loads(f.read())
                count = _generate_stubs_from_snapshot(data, be_name, stub_path)
            except Exception:
                count = 0

        if count == 0:
            count = _generate_stubs_from_live_module(fw, be_name, stub_path)

        # Copy snapshot_stubs.pyi to stub.pyi
        shutil.copyfile(stub_path, alt_stub_path)
        print(f"Generated {stub_path} and {alt_stub_path} ({count} stubs).")


def main() -> None:
    """Entry point for stub generation."""
    generate_stubs()


if __name__ == "__main__":
    main()
