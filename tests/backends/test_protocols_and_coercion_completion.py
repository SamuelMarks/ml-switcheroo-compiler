"""Completion unit tests for backend protocols, coercion helpers, and generator support utilities."""

from __future__ import annotations

import ast
import pathlib
import types
import typing
from collections.abc import Mapping

import libcst as cst
import numpy as np
import pytest
from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends._base_generator_support import _extract_node_attributes
from ml_switcheroo_compiler.backends._module_loader import (
    _EXPORT_SINK_VAR,
    _extract_stmt_names,
    record_exported_symbols,
)
from ml_switcheroo_compiler.backends.eager_registry.coercion import (
    coerce_array_list,
    coerce_bool,
    coerce_bytes,
    coerce_float,
    coerce_float_list,
    coerce_int,
    coerce_numeric,
    coerce_opt_callable,
    unpack_pair,
    unpack_quad,
    unpack_triple,
)
from ml_switcheroo_compiler.backends.eager_registry.protocols import (
    EagerBoundCallable,
    EagerRegisteredCallable,
    EagerTensorProtocol,
)
from ml_switcheroo_compiler.backends.eager_registry.registry_core import (
    EagerOpRegistry,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendArg, BackendOperand
from ml_switcheroo_compiler.backends.generator_mixins import EagerExecutionMixin
from ml_switcheroo_compiler.backends.hardware_config_models import (
    HardwareTemplatesConfig,
    _build_hardware_templates,
    _load_yaml_file_into_config,
)
from ml_switcheroo_compiler.backends.ir_to_ast import _emit_call_args
from ml_switcheroo_compiler.backends.protocols import (
    _MISSING_BACKEND_ATTR,
    BackendModuleProtocol,
    BackendNamespaceProtocol,
    EagerBackendAdapter,
    EagerCallableProtocol,
    StaticEagerBackendAdapter,
    _MissingBackendAttrSentinel,
    adapt_backend_module,
)
from ml_switcheroo_compiler.core.shape import CallableObjectProtocol, VariadicCallableProtocol
from ml_switcheroo_compiler.ir.shape_system import SymInt


class _NullSvdModule(types.ModuleType):
    """Module subclass exposing a `None` `svd` attribute for symbol traversal edge cases."""

    def __init__(self, name: str) -> None:
        """Initialize the module with `svd` explicitly set to `None`.

        Args:
            name (str): Module identifier.
        """
        super().__init__(name)
        self.svd: EagerTensorProtocol | None = None


def _sample_ones_op(*args: types.ModuleType | BackendOperand, **kwargs: BackendOperand) -> EagerTensorProtocol:
    """Delegate `ones` creation to numpy's eager backend adapter for binding tests.

    Args:
        *args (types.ModuleType | BackendOperand): Positional shape arguments.
        **kwargs (BackendOperand): Optional keyword dtype arguments.

    Returns:
        EagerTensorProtocol: Constructed ones tensor.
    """
    np_adapter = adapt_backend_module(np, backend_name="numpy")
    shape_arg = args[0] if args else (1,)
    shape = shape_arg if isinstance(shape_arg, tuple) else (1,)
    clean_shape = tuple(int(d) for d in shape if isinstance(d, int))
    raw_dtype = kwargs.get("dtype")
    dtype_val = raw_dtype if isinstance(raw_dtype, str) else None
    return np_adapter.ones(clean_shape, dtype=dtype_val)


class _DelegatingInternalAdapter:
    """Custom `EagerBackendAdapter` implementation with an internal compiler `__module__` prefix."""

    __module__ = "ml_switcheroo_compiler.backends.protocols"

    def __init__(self, inner: StaticEagerBackendAdapter) -> None:
        """Wrap `inner` for delegating all `EagerBackendAdapter` protocol methods.

        Args:
            inner (StaticEagerBackendAdapter): Backing static adapter instance.
        """
        self._inner = inner

    @property
    def backend_name(self) -> str:
        """Return canonical backend identifier.

        Returns:
            str: Canonical backend name.
        """
        return self._inner.backend_name

    def bound_operations(self) -> Mapping[str, EagerBoundCallable]:
        """Return explicitly bound operations.

        Returns:
            Mapping[str, EagerBoundCallable]: Bound operation callables.
        """
        return self._inner.bound_operations()

    def bound_dtypes(self) -> Mapping[str, BackendArg]:
        """Return explicitly bound dtypes.

        Returns:
            Mapping[str, BackendArg]: Bound dtype mapping.
        """
        return self._inner.bound_dtypes()

    def bound_submodules(self) -> Mapping[str, BackendModuleProtocol | types.ModuleType]:
        """Return explicitly bound submodules.

        Returns:
            Mapping[str, BackendModuleProtocol | types.ModuleType]: Bound submodule mapping.
        """
        return self._inner.bound_submodules()

    def get_op(self, name: str) -> EagerBoundCallable | None:
        """Look up bound operation `name`.

        Args:
            name (str): Operation identifier.

        Returns:
            EagerBoundCallable | None: Bound operation or `None`.
        """
        return self._inner.get_op(name)

    def require_op(self, name: str) -> EagerBoundCallable:
        """Require bound operation `name`.

        Args:
            name (str): Operation identifier.

        Returns:
            EagerBoundCallable: Bound operation callable.
        """
        return self._inner.require_op(name)

    def has_op(self, name: str) -> bool:
        """Check whether operation `name` is bound.

        Args:
            name (str): Operation identifier.

        Returns:
            bool: True if bound.
        """
        return self._inner.has_op(name)

    def get_dtype(self, name: str, default: BackendArg = None) -> BackendArg:
        """Look up bound dtype `name`.

        Args:
            name (str): Dtype identifier.
            default (BackendArg): Fallback value when absent.

        Returns:
            BackendArg: Bound dtype or `default`.
        """
        return self._inner.get_dtype(name, default=default)

    def has_dtype(self, name: str) -> bool:
        """Check whether dtype `name` is bound.

        Args:
            name (str): Dtype identifier.

        Returns:
            bool: True if bound.
        """
        return self._inner.has_dtype(name)

    def get_submodule(self, name: str) -> BackendModuleProtocol | types.ModuleType | None:
        """Look up bound submodule `name`.

        Args:
            name (str): Submodule identifier.

        Returns:
            BackendModuleProtocol | types.ModuleType | None: Bound submodule or `None`.
        """
        return self._inner.get_submodule(name)

    @typing.overload
    def resolve_symbol(self, dotted_path: str, default: None = ...) -> tuple[bool, EagerTensorProtocol | None]: ...

    @typing.overload
    def resolve_symbol(
        self,
        dotted_path: str,
        default: BackendOperand,
    ) -> tuple[bool, BackendOperand | types.ModuleType]: ...

    def resolve_symbol(
        self,
        dotted_path: str,
        default: BackendOperand = None,
    ) -> tuple[bool, BackendOperand | types.ModuleType]:
        """Resolve `dotted_path` via backing adapter.

        Args:
            dotted_path (str): Symbol name or dotted path.
            default (BackendOperand): Fallback value when unresolved.

        Returns:
            tuple[bool, BackendOperand | types.ModuleType]: Resolution tuple.
        """
        return self._inner.resolve_symbol(dotted_path, default)

    def has_symbol(self, dotted_path: str) -> bool:
        """Check whether `dotted_path` resolves.

        Args:
            dotted_path (str): Symbol name or dotted path.

        Returns:
            bool: True if symbol exists.
        """
        return self._inner.has_symbol(dotted_path)

    def require_symbol(self, dotted_path: str) -> EagerTensorProtocol:
        """Require resolved symbol `dotted_path`.

        Args:
            dotted_path (str): Symbol name or dotted path.

        Returns:
            EagerTensorProtocol: Resolved symbol.
        """
        return self._inner.require_symbol(dotted_path)

    @typing.overload
    def get_symbol(self, dotted_path: str) -> EagerTensorProtocol: ...

    @typing.overload
    def get_symbol(self, dotted_path: str, default: bool) -> bool: ...

    @typing.overload
    def get_symbol(self, dotted_path: str, default: int) -> int: ...

    @typing.overload
    def get_symbol(self, dotted_path: str, default: float) -> float: ...

    @typing.overload
    def get_symbol(self, dotted_path: str, default: str) -> str: ...

    @typing.overload
    def get_symbol(self, dotted_path: str, default: list[BackendArg]) -> EagerTensorProtocol | list[BackendArg]: ...

    @typing.overload
    def get_symbol(
        self, dotted_path: str, default: dict[str, BackendArg]
    ) -> EagerTensorProtocol | dict[str, EagerTensorProtocol]: ...

    @typing.overload
    def get_symbol(
        self, dotted_path: str, default: tuple[BackendArg, ...]
    ) -> EagerTensorProtocol | tuple[BackendArg, ...]: ...

    @typing.overload
    def get_symbol(
        self,
        dotted_path: str,
        default: EagerTensorProtocol
        | BackendModuleProtocol
        | BackendNamespaceProtocol
        | EagerCallableProtocol
        | CallableObjectProtocol
        | VariadicCallableProtocol
        | type,
    ) -> EagerTensorProtocol: ...

    @typing.overload
    def get_symbol(self, dotted_path: str, default: None) -> EagerTensorProtocol | BackendArg: ...

    @typing.overload
    def get_symbol(self, dotted_path: str, default: BackendOperand) -> EagerTensorProtocol | BackendArg: ...

    def get_symbol(
        self,
        dotted_path: str,
        default: BackendOperand | _MissingBackendAttrSentinel = _MISSING_BACKEND_ATTR,
    ) -> BackendOperand | types.ModuleType:
        """Return resolved symbol `dotted_path` or `default`.

        Args:
            dotted_path (str): Symbol name or dotted path.
            default (BackendOperand | _MissingBackendAttrSentinel): Optional fallback value.

        Returns:
            BackendOperand | types.ModuleType: Resolved symbol or fallback.
        """
        if isinstance(default, _MissingBackendAttrSentinel):
            return self._inner.get_symbol(dotted_path)
        return self._inner.get_symbol(dotted_path, default)

    def asarray(self, data: BackendArg, dtype: BackendArg = None) -> EagerTensorProtocol:
        """Convert `data` into a backend tensor.

        Args:
            data (BackendArg): Input scalar or sequence.
            dtype (BackendArg): Optional target dtype.

        Returns:
            EagerTensorProtocol: Converted tensor.
        """
        return self._inner.asarray(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: BackendArg = None) -> EagerTensorProtocol:
        """Create a zero-filled tensor.

        Args:
            shape (tuple[int, ...]): Target shape.
            dtype (BackendArg): Optional target dtype.

        Returns:
            EagerTensorProtocol: Zero tensor.
        """
        return self._inner.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: BackendArg = None) -> EagerTensorProtocol:
        """Create a one-filled tensor.

        Args:
            shape (tuple[int, ...]): Target shape.
            dtype (BackendArg): Optional target dtype.

        Returns:
            EagerTensorProtocol: Ones tensor.
        """
        return self._inner.ones(shape, dtype=dtype)


class _DelegatingExternalAdapter(_DelegatingInternalAdapter):
    """Adapter subclass keeping a test-suite `__module__` outside `ml_switcheroo_compiler`."""


class _LocalOnlyEagerRegistry(EagerOpRegistry):
    """Registry subclass returning only locally registered operations from `get`."""

    def get(self, op_type: str) -> EagerRegisteredCallable | None:
        """Look up `op_type` strictly in the local dictionary without global fallback.

        Args:
            op_type (str): Canonical operation identifier.

        Returns:
            EagerRegisteredCallable | None: Locally registered callable or `None`.
        """
        return self._registry.get(op_type)


class _PlainGeneratorMixin(EagerExecutionMixin):
    """Generator mixin subclass without `get_module`."""


class _WorkingGeneratorMixin(EagerExecutionMixin):
    """Generator mixin subclass whose `get_module` returns `numpy`."""

    @classmethod
    def get_module(cls) -> types.ModuleType:
        """Return the `numpy` module.

        Returns:
            types.ModuleType: Loaded numpy module.
        """
        return np


class _FailingGeneratorMixin(EagerExecutionMixin):
    """Generator mixin subclass whose `get_module` raises `ImportError`."""

    @classmethod
    def get_module(cls) -> types.ModuleType:
        """Simulate an unavailable optional runtime backend.

        Returns:
            types.ModuleType: Never returns normally.

        Raises:
            ImportError: Always raised to exercise fallback adapter creation.
        """
        raise ImportError("simulated missing backend")


def test_static_eager_backend_adapter_metadata_and_tables() -> None:
    """Verify `StaticEagerBackendAdapter` properties, bound mappings, dtypes, and submodules."""
    adapter = StaticEagerBackendAdapter(
        backend_name="custom_bk",
        ops={"ones": _sample_ones_op},
        dtypes={"fp32": "float32_token"},
        submodules={"linalg": np.linalg},
        module=None,
    )
    assert adapter.backend_name == "custom_bk"
    assert adapter.__name__ == "custom_bk"

    mod_adapter = StaticEagerBackendAdapter(backend_name="np_bk", module=np)
    assert mod_adapter.__name__ == "numpy"

    ops_copy = adapter.bound_operations()
    assert "ones" in ops_copy
    assert adapter.bound_dtypes() == {"fp32": "float32_token"}
    assert adapter.bound_submodules() == {"linalg": np.linalg}

    assert adapter.get_dtype("fp32") == "float32_token"
    assert adapter.get_dtype("missing_dt", default="fallback_dt") == "fallback_dt"
    assert mod_adapter.get_dtype("float32") is np.float32
    assert mod_adapter.get_dtype("definitely_missing_dt", default="none_dt") == "none_dt"

    assert adapter.has_dtype("fp32") is True
    assert adapter.has_dtype("missing_dt") is False
    assert mod_adapter.has_dtype("float64") is True
    assert mod_adapter.has_dtype("definitely_missing_dt") is False

    assert adapter.get_submodule("linalg") is np.linalg
    assert adapter.get_submodule("fft") is None
    assert mod_adapter.get_submodule("fft") is np.fft
    assert mod_adapter.get_submodule("definitely_missing_sub") is None


def test_static_eager_backend_adapter_symbol_resolution_and_requirements() -> None:
    """Verify single-key and dotted symbol resolution, requirement errors, `ones`, and adaptation."""
    np_mod_adapter = StaticEagerBackendAdapter(backend_name="numpy", module=np)
    proto_sub_tensor: EagerTensorProtocol = np_mod_adapter.ones((2,))
    assert tuple(proto_sub_tensor.shape) == (2,)
    typed_ones: EagerTensorProtocol = np_mod_adapter.ones((2, 3), dtype=np.float32)
    assert tuple(typed_ones.shape) == (2, 3)

    null_svd_mod = _NullSvdModule("linalg")
    plain_mod = types.ModuleType("plain_sub")
    adapter = StaticEagerBackendAdapter(
        backend_name="sym_bk",
        ops={"ones": _sample_ones_op},
        dtypes={"fp16": "float16_token"},
        submodules={
            "linalg": np.linalg,
            "tensor_ns": proto_sub_tensor,
            "plain_mod": plain_mod,
            "null_linalg": null_svd_mod,
        },
        module=None,
    )

    found_op, val_op = adapter._resolve_single_key("ones", None)
    assert found_op is True
    assert callable(val_op)

    found_dt, val_dt = adapter._resolve_single_key("fp16", None)
    assert (found_dt, val_dt) == (True, "float16_token")

    found_sub, val_sub = adapter._resolve_single_key("linalg", None)
    assert (found_sub, val_sub) == (True, np.linalg)

    found_miss, val_miss = adapter._resolve_single_key("unknown_key", "def_val")
    assert (found_miss, val_miss) == (False, "def_val")

    assert adapter.resolve_symbol("", default="empty_def") == (False, "empty_def")
    assert adapter.resolve_symbol("...", default="dots_def") == (False, "dots_def")
    assert adapter.resolve_symbol("unregistered_root.svd", default="no_mod") == (False, "no_mod")
    assert adapter.resolve_symbol("null_linalg.svd.u", default="null_hop") == (False, "null_hop")
    assert adapter.resolve_symbol("linalg.missing_leaf_xyz", default="no_leaf") == (False, "no_leaf")

    found_svd, resolved_svd = adapter.resolve_symbol("linalg.svd")
    assert found_svd is True
    assert id(resolved_svd) == id(np.linalg.svd)

    assert adapter._require_single_live_symbol("tensor_ns") is proto_sub_tensor
    with pytest.raises(AttributeError, match="plain_mod"):
        adapter._require_single_live_symbol("plain_mod")
    with pytest.raises(AttributeError, match="absent_key"):
        adapter._require_single_live_symbol("absent_key")

    with pytest.raises(AttributeError):
        adapter.require_symbol("")
    assert id(adapter.require_symbol("linalg.svd")) == id(np.linalg.svd)
    assert id(np_mod_adapter.require_symbol("linalg.svd")) == id(np.linalg.svd)
    with pytest.raises(AttributeError, match="unregistered_root.svd"):
        adapter.require_symbol("unregistered_root.svd")
    with pytest.raises(AttributeError, match="null_linalg.svd.u"):
        adapter.require_symbol("null_linalg.svd.u")
    with pytest.raises(AttributeError, match="linalg.missing_leaf_xyz"):
        adapter.require_symbol("linalg.missing_leaf_xyz")
    with pytest.raises(AttributeError, match="null_linalg.svd"):
        adapter.require_symbol("null_linalg.svd")

    with pytest.raises(AttributeError, match="completely_missing_symbol"):
        adapter.get_symbol("completely_missing_symbol")
    assert adapter.get_symbol("completely_missing_symbol", default=77) == 77
    assert id(np_mod_adapter.get_symbol("add")) == id(np.add)


def test_adapt_backend_module_passthrough_and_wrapping() -> None:
    """Verify `adapt_backend_module` pass-through for internal adapters and wrapping for external ones."""
    adapter = StaticEagerBackendAdapter(backend_name="numpy", module=np)
    assert adapt_backend_module(adapter) is adapter

    internal_proto: EagerBackendAdapter = _DelegatingInternalAdapter(adapter)
    assert isinstance(internal_proto, EagerBackendAdapter)
    assert adapt_backend_module(internal_proto) is internal_proto

    external_proto: EagerBackendAdapter = _DelegatingExternalAdapter(adapter)
    wrapped_external = adapt_backend_module(external_proto, backend_name="wrapped_ext")
    assert isinstance(wrapped_external, StaticEagerBackendAdapter)
    assert wrapped_external.backend_name == "wrapped_ext"


def test_coercion_helpers_edge_branches() -> None:
    """Verify scalar, byte-buffer, float-list, tuple-unpacking, callable, and array-list coercions."""
    assert coerce_int(None, default=9) == 9
    assert coerce_int(True) == 1
    assert coerce_int(False) == 0

    assert coerce_float(None, default=3.5) == 3.5
    assert coerce_bool(None, default=True) is True
    assert coerce_bool(None, default=False) is False

    assert coerce_bytes("switcheroo") == b"switcheroo"
    np_adapter = adapt_backend_module(np, backend_name="numpy")
    uint8_buf = np_adapter.asarray([65, 66, 67], dtype=np.uint8)
    assert coerce_bytes(uint8_buf) == b"ABC"
    assert coerce_bytes([68, 69]) == b"DE"
    with pytest.raises(TypeError, match="Expected bytes-convertible value"):
        coerce_bytes(1.25)

    assert coerce_float_list(None) == []
    assert coerce_float_list(5) == [5.0]
    assert coerce_float_list(2.25) == [2.25]
    assert coerce_float_list([1, 2.5]) == [1.0, 2.5]
    with pytest.raises(TypeError, match="Expected iterable of floats"):
        coerce_float_list("not_floats")
    with pytest.raises(TypeError, match="Expected iterable of floats"):
        coerce_float_list(True)

    assert unpack_pair([10, 20]) == (10, 20)
    with pytest.raises(ValueError, match="Expected 2 elements"):
        unpack_pair([10, 20, 30])

    assert unpack_triple([1, 2, 3]) == (1, 2, 3)
    with pytest.raises(TypeError, match="Expected 3-element indexable sequence"):
        unpack_triple(42)
    with pytest.raises(ValueError, match="Expected 3 elements"):
        unpack_triple([1, 2])

    assert unpack_quad([1, 2, 3, 4]) == (1, 2, 3, 4)
    with pytest.raises(TypeError, match="Expected 4-element indexable sequence"):
        unpack_quad(42)
    with pytest.raises(ValueError, match="Expected 4 elements"):
        unpack_quad([1, 2, 3])

    assert coerce_opt_callable(None) is None
    assert coerce_opt_callable(_sample_ones_op) is _sample_ones_op
    with pytest.raises(TypeError, match="Expected callable or None"):
        coerce_opt_callable(123)

    coerced_true = coerce_numeric(True)
    assert coerced_true == 1
    assert type(coerced_true) is int

    arr_list = coerce_array_list(np, [[1.0, 2.0], [3.0, 4.0]])
    assert len(arr_list) == 2
    assert tuple(arr_list[0].shape) == (2,)
    assert tuple(arr_list[1].shape) == (2,)


def test_registry_core_global_fallback_when_subclass_get_returns_none() -> None:
    """Verify `EagerOpRegistry.dispatch` global registry fallback on line 94."""
    op_name = "_TempGlobalDispatchFallbackOp94"
    local_only = _LocalOnlyEagerRegistry()
    global_eager_registry.register(op_name)(_sample_ones_op)
    try:
        assert local_only.get(op_name) is None
        result = local_only.dispatch(op_name, (4,))
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

        with pytest.raises(ValueError, match="not found in registry"):
            local_only.dispatch("_CompletelyUnknownOpForLine94")
    finally:
        global_eager_registry._registry.pop(op_name, None)


def test_base_generator_support_and_module_loader_branches() -> None:
    """Verify `_extract_node_attributes`, `record_exported_symbols`, and `_extract_stmt_names`."""
    assert _extract_node_attributes("plain_string_arg") == {}
    assert _extract_node_attributes(None) == {}
    ir_node = IRNode(id="add_0", op_type="Add", attributes={"alpha": 2})
    assert _extract_node_attributes(ir_node) == {"alpha": 2}

    reset_none = _EXPORT_SINK_VAR.set(None)
    try:
        record_exported_symbols({"ignored_symbol": 123})
        assert _EXPORT_SINK_VAR.get() is None
    finally:
        _EXPORT_SINK_VAR.reset(reset_none)

    sink_box: dict[str, BackendArg] = {}
    reset_sink = _EXPORT_SINK_VAR.set(sink_box)
    try:
        record_exported_symbols({"captured_symbol": 456})
        assert sink_box == {"captured_symbol": 456}
    finally:
        _EXPORT_SINK_VAR.reset(reset_sink)

    ann_stmt = ast.parse("count: int = 10").body[0]
    expr_stmt = ast.parse("count + 1").body[0]
    attr_ann_stmt = ast.parse("holder.count: int = 10").body[0]
    assert _extract_stmt_names(ann_stmt) == ["count"]
    assert _extract_stmt_names(expr_stmt) == []
    assert _extract_stmt_names(attr_ann_stmt) == []


def test_generator_mixins_get_eager_adapter_variants() -> None:
    """Verify `EagerExecutionMixin.get_eager_adapter` across plain, working, and failing subclasses."""
    plain_adapter = _PlainGeneratorMixin.get_eager_adapter()
    assert isinstance(plain_adapter, StaticEagerBackendAdapter)
    assert plain_adapter.backend_name == "_plaingeneratormixin"

    working_adapter = _WorkingGeneratorMixin.get_eager_adapter()
    assert working_adapter.backend_name == "_workinggeneratormixin"
    assert working_adapter.has_op("add") is True

    failing_adapter = _FailingGeneratorMixin.get_eager_adapter()
    assert isinstance(failing_adapter, StaticEagerBackendAdapter)
    assert failing_adapter.backend_name == "_failinggeneratormixin"
    assert failing_adapter.has_op("add") is False


def test_hardware_config_models_dir_without_monolith_and_invalid_templates_list(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `_build_hardware_templates` missing `yaml_path` branch and non-dict `templates` handling.

    Args:
        tmp_path (pathlib.Path): Temporary directory fixture provided by pytest.
    """
    ops_dir = tmp_path / "ops_templates"
    ops_dir.mkdir()
    (ops_dir / "Add.yaml").write_text("Add:\n  body: out = a + b;\n", encoding="utf-8")
    missing_monolith = tmp_path / "does_not_exist.yaml"

    built_cfg = _build_hardware_templates(str(ops_dir), str(missing_monolith), backend=None)
    assert "Add" in built_cfg.templates

    invalid_dir_yaml = tmp_path / "bad_templates.yaml"
    invalid_dir_yaml.write_text("templates:\n  - not_a_dict\n", encoding="utf-8")
    target_cfg = HardwareTemplatesConfig(templates={})
    loaded = _load_yaml_file_into_config(str(invalid_dir_yaml), target_cfg, is_dir=True)
    assert loaded is False
    assert target_cfg.templates == {}


def test_ir_to_ast_emit_call_args_non_primitive_attributes() -> None:
    """Verify `_emit_call_args` stringifies identifier-like and non-identifier non-primitive attributes."""
    node = IRNode(
        id="op_0",
        op_type="CustomOp",
        inputs=["x"],
        attributes={
            "sym_dim": SymInt("batch_size"),
            "none_flag": None,
            "dims_list": [2, 4],
        },
    )
    emitted_args = _emit_call_args(node, target_framework="unknown_fw")
    kw_names: dict[str, str] = {}
    for arg in emitted_args:
        if arg.keyword is not None and isinstance(arg.value, cst.Name):
            kw_names[arg.keyword.value] = arg.value.value

    assert kw_names["sym_dim"] == "batch_size"
    assert kw_names["none_flag"] == "None"
    assert kw_names["dims_list"] == "custom_val"
