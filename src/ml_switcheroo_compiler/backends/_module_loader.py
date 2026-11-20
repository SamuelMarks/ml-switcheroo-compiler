"""File-backed generated Python kernel module loader replacing raw ``exec()``."""

from __future__ import annotations

import ast
import contextvars
import importlib.util
import keyword
import linecache
import os
import sys
import tempfile
import typing
import uuid
from collections.abc import Mapping, MutableMapping

from ml_switcheroo_compiler.core.errors import CompilationError

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
    from ml_switcheroo_compiler.core.shape import OpValue

_NamespaceValT = typing.TypeVar("_NamespaceValT", "BackendArg", "OpValue")
_SEED_VAR: contextvars.ContextVar[Mapping[str, BackendArg | OpValue] | None] = contextvars.ContextVar(
    "_SEED_VAR",
    default=None,
)
_EXPORT_SINK_VAR: contextvars.ContextVar[dict[str, BackendArg] | None] = contextvars.ContextVar(
    "_EXPORT_SINK_VAR",
    default=None,
)


def get_seed_symbol(name: str) -> BackendArg | OpValue:
    """Retrieve a pre-seeded symbol for a generated kernel module during execution.

    Args:
        name (str): Pre-seeded global variable identifier.

    Returns:
        BackendArg | OpValue: Bound pre-seeded value, or ``None`` if unset.
    """
    active = _SEED_VAR.get()
    return active[name] if active is not None and name in active else None


def record_exported_symbols(symbols: Mapping[str, BackendArg]) -> None:
    """Record top-level module symbols produced by an executed generated kernel file.

    Args:
        symbols (Mapping[str, BackendArg]): Exported symbol mapping from generated module.
    """
    sink = _EXPORT_SINK_VAR.get()
    if sink is not None:
        sink.update(symbols)


def _extract_stmt_names(node: ast.stmt) -> list[str]:
    """Extract defined symbol identifiers from a single top-level AST statement.

    Args:
        node (ast.stmt): Top-level AST statement node.

    Returns:
        list[str]: Defined symbol names on ``node``.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [
            sub.id
            for target in node.targets
            for sub in ast.walk(target)
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
        ]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [
            alias.asname or alias.name.split(".")[0]
            for alias in node.names
            if (alias.asname or alias.name.split(".")[0]) != "*"
        ]
    return []


def _collect_defined_names(source_code: str) -> list[str]:
    """Extract top-level defined identifier names from ``source_code`` via ``ast.parse``.

    Args:
        source_code (str): Prepared Python source code.

    Returns:
        list[str]: Sorted non-dunder top-level symbol names.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    names = {n for stmt in tree.body for n in _extract_stmt_names(stmt)}
    return sorted(n for n in names if n.isidentifier() and not keyword.iskeyword(n) and not n.startswith("__"))


def _inject_seed_header(
    source_code: str,
    namespace: Mapping[str, _NamespaceValT] | None,
) -> str:
    """Insert pre-seeded global bindings and export-recording epilogue.

    Args:
        source_code (str): Original generated Python code.
        namespace (Mapping[str, _NamespaceValT] | None): Optional pre-seeded symbol mapping.

    Returns:
        str: Augmented Python source code.
    """
    valid_keys = (
        [k for k in namespace if k.isidentifier() and not keyword.iskeyword(k) and not k.startswith("__")]
        if namespace
        else []
    )
    seed_lines: list[str] = []
    if valid_keys:
        seed_lines.append("from ml_switcheroo_compiler.backends._module_loader import get_seed_symbol as _sw_get_seed")
        seed_lines.extend(f"{k} = _sw_get_seed({k!r})" for k in valid_keys)

    src_lines = source_code.splitlines()
    insert_idx = 0
    for idx, line in enumerate(src_lines):
        if line.strip().startswith("from __future__ import"):
            insert_idx = idx + 1
    merged = src_lines[:insert_idx] + seed_lines + src_lines[insert_idx:]
    body_text = os.linesep.join(merged) + os.linesep

    defined = [n for n in _collect_defined_names(body_text) if n != "_sw_get_seed"]
    if not defined:
        return body_text
    entries = ", ".join(f"{n!r}: {n}" for n in defined)
    epilogue = (
        f"from ml_switcheroo_compiler.backends._module_loader import record_exported_symbols as _sw_rec{os.linesep}"
        f"_sw_rec({{{entries}}}){os.linesep}"
    )
    return body_text + epilogue


def load_generated_code(
    source_code: str,
    namespace: MutableMapping[str, _NamespaceValT] | None = None,
    module_name: str = "_switcheroo_generated_kernel",
) -> dict[str, BackendArg]:
    """Write generated Python code to a temporary file and load it via ``importlib.util``.

    Caches the generated source in ``linecache`` so stack traces and JIT compilers
    (such as Numba ``@njit`` / ``@vectorize``) can introspect source lines after
    temporary directory cleanup.

    Args:
        source_code (str): Generated Python module source string.
        namespace (MutableMapping[str, _NamespaceValT] | None): Optional namespace pre-seeded
            into the module globals and updated in place after module execution.
        module_name (str): Synthetic module identifier used for ``spec_from_file_location``.

    Returns:
        dict[str, BackendArg]: Executed module symbol dictionary.

    Raises:
        CompilationError: If ``importlib.util.spec_from_file_location`` fails to create a loader.
    """
    unique_mod_name = f"{module_name}_{uuid.uuid4().hex[:8]}"
    prepared_source = _inject_seed_header(source_code, namespace)
    with tempfile.TemporaryDirectory(prefix="switcheroo_codegen_") as tmp_dir:
        file_path = os.path.join(tmp_dir, f"{unique_mod_name}.py")
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(prepared_source)

        linecache.cache[file_path] = (
            len(prepared_source),
            None,
            prepared_source.splitlines(keepends=True),
            file_path,
        )

        spec = importlib.util.spec_from_file_location(unique_mod_name, file_path)
        if spec is None or spec.loader is None:
            raise CompilationError(f"Unable to construct module spec for generated kernel {module_name!r}.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_mod_name] = module
        exported: dict[str, BackendArg] = {}
        seed_token = _SEED_VAR.set(namespace)
        sink_token = _EXPORT_SINK_VAR.set(exported)
        try:
            spec.loader.exec_module(module)
        finally:
            _EXPORT_SINK_VAR.reset(sink_token)
            _SEED_VAR.reset(seed_token)

        if namespace is not None:
            for k, v in exported.items():
                namespace[k] = v
        return exported


__all__ = ["get_seed_symbol", "load_generated_code", "record_exported_symbols"]
