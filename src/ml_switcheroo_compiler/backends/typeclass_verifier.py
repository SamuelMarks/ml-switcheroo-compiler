"""Static AST, capability trait, and curated YAML verifier for compiler backends.

Enforces Haskell-style capability proofs across registered `@register_backend` classes,
live `CoreEagerTypeclass` dictionaries, and curated declarative YAML op mappings.
"""

from __future__ import annotations

import ast
import os
import typing

import pydantic
import yaml
from ml_switcheroo_ir.typeclasses import Invalid, Semigroup, Valid, Validation, zip_validation3

from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand
from ml_switcheroo_compiler.backends.mapping_loader import OpMappingSchema
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.backends.typeclasses import (
    CORE_EAGER_REQUIRED_OPS,
    CapabilityKind,
    CoreEagerTypeclass,
)

DENSE_CORE_EAGER_BACKENDS: tuple[str, ...] = (
    "numpy",
    "pure_python",
    "pytorch",
    "jax",
    "tensorflow",
    "keras",
    "mlx",
    "dask",
)
_ALLOWED_MAPPING_KEYS: frozenset[str] = frozenset(OpMappingSchema.model_fields.keys())


@typing.runtime_checkable
class _HasTypeclassDictProtocol(typing.Protocol):
    """Protocol for eager adapters exposing an optional `.typeclass_dict` property."""

    @property
    def typeclass_dict(self) -> CoreEagerTypeclass[BackendOperand] | BackendOperand | None:
        """Return bound `CoreEagerTypeclass` dictionary or `None`.

        Returns:
            CoreEagerTypeclass[BackendOperand] | BackendOperand | None: Bound typeclass dictionary or `None`.
        """
        ...


def _is_register_backend_decorator(dec: ast.expr) -> bool:
    """Return True when `dec` is a `@register_backend(...)` call expression.

    Args:
        dec (ast.expr): Decorator AST expression.

    Returns:
        bool: True if `dec` invokes `register_backend`.
    """
    if not isinstance(dec, ast.Call):
        return False
    func = dec.func
    if isinstance(func, ast.Name):
        return func.id == "register_backend"
    if isinstance(func, ast.Attribute):
        return func.attr == "register_backend"
    return False


def _collect_class_method_names(cls_node: ast.ClassDef) -> set[str]:
    """Return all function and method names defined directly on `cls_node`.

    Args:
        cls_node (ast.ClassDef): Backend generator class AST node.

    Returns:
        set[str]: Set of method names defined in `cls_node.body`.
    """
    return {stmt.name for stmt in cls_node.body if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _check_registered_class_ast(path: str, cls_node: ast.ClassDef) -> list[str]:
    """Verify AST-level capability invariants on a `@register_backend` class definition.

    Args:
        path (str): Source file path.
        cls_node (ast.ClassDef): Decorated class AST node.

    Returns:
        list[str]: Violation messages, or empty list when compliant.
    """
    violations: list[str] = []
    methods = _collect_class_method_names(cls_node)
    if not cls_node.bases and "generate" not in methods:
        violations.append(
            f"{path}:{cls_node.lineno}: backend '{cls_node.name}' lacks CodeEmitterTypeclass 'generate' implementation"
        )
    if "_compile_aot_impl" not in methods and "compile_aot" not in methods:
        violations.append(
            f"{path}:{cls_node.lineno}: backend '{cls_node.name}' lacks AOTCompilerTypeclass '_compile_aot_impl'"
        )
    return violations


def verify_backend_ast_capabilities(backends_dir: str) -> list[str]:
    """Scan Python modules under `backends_dir` and verify `@register_backend` class obligations.

    Args:
        backends_dir (str): Path to `src/ml_switcheroo_compiler/backends` or test fixture directory.

    Returns:
        list[str]: Sorted AST capability violation messages.
    """
    if not os.path.isdir(backends_dir):
        return [f"{backends_dir}:1: backends directory does not exist"]
    violations: list[str] = []
    for root, _, files in sorted(os.walk(backends_dir)):
        if "adapter_bindings" in root:
            continue
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=fpath)
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and any(
                    _is_register_backend_decorator(d) for d in node.decorator_list
                ):
                    violations.extend(_check_registered_class_ast(fpath, node))
    return violations


def _verify_single_eager_backend(name: str) -> list[str]:
    """Verify `CoreEagerTypeclass` obligations for a single `CORE_EAGER` backend.

    Args:
        name (str): Canonical backend identifier.

    Returns:
        list[str]: Violation messages (empty when compliant or optional dependency is absent).
    """
    violations: list[str] = []
    try:
        adapter = BackendRegistry.get(name).get_eager_adapter()
    except ImportError:
        return []
    bound_keys = set(adapter.bound_operations().keys())
    tc_candidate = adapter.typeclass_dict if isinstance(adapter, _HasTypeclassDictProtocol) else None
    if tc_candidate is None:
        violations.append(f"{name}: CORE_EAGER backend missing verified typeclass_dict")
    elif not isinstance(tc_candidate, CoreEagerTypeclass):
        violations.append(f"{name}: typeclass_dict does not conform to CoreEagerTypeclass")
    missing = sorted(CORE_EAGER_REQUIRED_OPS - bound_keys)
    if missing:
        violations.append(f"{name}: CoreEagerTypeclass adapter missing ops: {', '.join(missing)}")
    return violations


def verify_live_typeclass_instances(
    backend_names: tuple[str, ...] = DENSE_CORE_EAGER_BACKENDS,
) -> list[str]:
    """Verify that live registered backends satisfy their declared `CapabilityKind` contracts.

    Args:
        backend_names (tuple[str, ...]): Registered backend names to inspect in-process.

    Returns:
        list[str]: Violation messages for missing capabilities or incomplete typeclass dictionaries.
    """
    violations: list[str] = []
    for name in backend_names:
        try:
            gen_cls = BackendRegistry.get(name)
        except (ImportError, ValueError):
            continue
        caps = gen_cls.declared_capabilities()
        if CapabilityKind.CODE_EMIT not in caps:
            violations.append(f"{name}: missing CapabilityKind.CODE_EMIT declaration")
        if CapabilityKind.AOT_COMPILE not in caps:
            violations.append(f"{name}: missing CapabilityKind.AOT_COMPILE declaration")
        if CapabilityKind.CORE_EAGER in caps:
            violations.extend(_verify_single_eager_backend(name))
    return violations


def _validate_single_mapping_file(fpath: str) -> list[str]:
    """Validate a single curated `mappings/*.yaml` file against `OpMappingSchema`.

    Args:
        fpath (str): Path to a backend YAML mapping file.

    Returns:
        list[str]: Schema violation messages for `fpath`.
    """
    with open(fpath, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        return [f"{fpath}:1: expected YAML mapping dictionary"]
    unknown_keys = sorted(set(str(k) for k in raw) - _ALLOWED_MAPPING_KEYS)
    if unknown_keys:
        return [f"{fpath}:1: unknown OpMappingSchema keys: {', '.join(unknown_keys)}"]
    try:
        parsed = OpMappingSchema.model_validate(raw)
    except pydantic.ValidationError as exc:
        return [f"{fpath}:1: invalid OpMappingSchema: {exc.errors()[0]['msg']}"]
    if parsed.operation is not None and not parsed.operation.strip():
        return [f"{fpath}:1: 'operation' field must not be blank"]
    return []


def verify_curated_yaml_mappings(backends_dir: str) -> list[str]:
    """Validate retained curated `mappings/*.yaml` files across all backend subdirectories.

    Args:
        backends_dir (str): Path to `src/ml_switcheroo_compiler/backends` or test fixture directory.

    Returns:
        list[str]: Schema violation messages across curated YAML mappings.
    """
    if not os.path.isdir(backends_dir):
        return [f"{backends_dir}:1: backends directory does not exist"]
    violations: list[str] = []
    for entry in sorted(os.listdir(backends_dir)):
        mappings_dir = os.path.join(backends_dir, entry, "mappings")
        if not os.path.isdir(mappings_dir):
            continue
        for fname in sorted(os.listdir(mappings_dir)):
            if fname.endswith(".yaml"):
                violations.extend(_validate_single_mapping_file(os.path.join(mappings_dir, fname)))
    return violations


def verify_all_typeclasses(
    backends_dir: str = "src/ml_switcheroo_compiler/backends",
    live_backends: tuple[str, ...] = DENSE_CORE_EAGER_BACKENDS,
) -> list[str]:
    """Execute AST capability verification, live typeclass checks, and curated YAML schema checks.

    Args:
        backends_dir (str): Root directory containing compiler backend packages.
        live_backends (tuple[str, ...]): Dense eager backends to verify live when installed.

    Returns:
        list[str]: Combined list of static verification violations (empty when 100% compliant).
    """
    violations: list[str] = []
    violations.extend(verify_backend_ast_capabilities(backends_dir))
    violations.extend(verify_live_typeclass_instances(live_backends))
    violations.extend(verify_curated_yaml_mappings(backends_dir))
    return violations


_VIOLATION_TUPLE_SG: Semigroup[tuple[str, ...]] = Semigroup(combine=lambda a, b: a + b)


def _to_unit_validation(errs: list[str]) -> Validation[tuple[str, ...], None]:
    """Convert a list of violation strings into ``Valid(None)`` or ``Invalid(tuple(errs))``.

    Args:
        errs (list[str]): Collected violation diagnostic strings.

    Returns:
        Validation[tuple[str, ...], None]: Applicative validation value.
    """
    if errs:
        return Invalid(tuple(errs))
    return Valid(None)


def validate_all_typeclasses_checked(
    backends_dir: str = "src/ml_switcheroo_compiler/backends",
    live_backends: tuple[str, ...] = DENSE_CORE_EAGER_BACKENDS,
) -> Validation[tuple[str, ...], tuple[str, ...]]:
    """Applicatively validate AST capabilities, live typeclass dictionaries, and YAML schemas without short-circuiting.

    Args:
        backends_dir (str): Root directory containing compiler backend packages.
        live_backends (tuple[str, ...]): Dense eager backends to verify live when installed.

    Returns:
        Validation[tuple[str, ...], tuple[str, ...]]: ``Valid(live_backends)`` or ``Invalid`` with all violations.
    """
    v_ast = _to_unit_validation(verify_backend_ast_capabilities(backends_dir))
    v_live = _to_unit_validation(verify_live_typeclass_instances(live_backends))
    v_yaml = _to_unit_validation(verify_curated_yaml_mappings(backends_dir))
    return zip_validation3(
        _VIOLATION_TUPLE_SG,
        v_ast,
        v_live,
        v_yaml,
        lambda _a, _b, _c: live_backends,
    )


__all__ = [
    "DENSE_CORE_EAGER_BACKENDS",
    "validate_all_typeclasses_checked",
    "verify_all_typeclasses",
    "verify_backend_ast_capabilities",
    "verify_curated_yaml_mappings",
    "verify_live_typeclass_instances",
]
