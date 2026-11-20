"""Strongly-typed structural protocols and eager backend adapters for backend modules and tensors."""

from __future__ import annotations

import types
import typing
from collections.abc import Mapping

from ml_switcheroo_compiler.backends.adapter_bindings import read_static_attr
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    BackendOperand,
    EagerBoundCallable,
    EagerCallableProtocol,
    EagerTensorProtocol,
    ShapeProtocol,
    coerce_float,
    coerce_int,
    coerce_shape,
    require_callable,
)
from ml_switcheroo_compiler.core.shape import (
    CallableLike,
    CallableObjectProtocol,
    VariadicCallableProtocol,
)


@typing.runtime_checkable
class _PydanticSchemaHandlerProtocol(typing.Protocol):
    """Protocol for Pydantic core schema handlers."""

    def generate_schema(self, source_type: BackendArg) -> BackendArg:
        """Generate Pydantic core schema for `source_type`.

        Args:
            source_type (BackendArg): Source type annotation.

        Returns:
            BackendArg: Generated core schema.
        """
        ...


@typing.runtime_checkable
class BackendNodeProtocol(typing.Protocol):
    """Structural protocol for backend IR and logical graph nodes."""

    id: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, BackendArg]
    shape_metadata: tuple[int | str, ...] | list[int | str] | None
    dtype_metadata: str | None
    metadata: dict[str, BackendArg]


@typing.runtime_checkable
class BackendGraphProtocol(typing.Protocol):
    """Structural protocol for backend IR and logical computation graphs."""

    nodes: dict[str, BackendNodeProtocol]
    inputs: list[str]
    outputs: list[str]
    metadata: dict[str, BackendArg]


BackendTensorProtocol = EagerTensorProtocol


class _MissingBackendAttrSentinel:
    """Sentinel type for omitted default argument in `get_symbol`."""


_MISSING_BACKEND_ATTR = _MissingBackendAttrSentinel()


class _BaseNsProto(typing.Protocol):
    """Base protocol for callable sub-namespaces."""

    abs: EagerCallableProtocol

    def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> EagerTensorProtocol:
        """Invoke when namespace attribute is itself a callable.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            EagerTensorProtocol: Computed tensor result.
        """
        ...


class BackendNamespaceProtocol(_BaseNsProto):
    """Protocol for sub-namespaces on a backend module (e.g., linalg, fft, random, special)."""


BackendModuleProtocol = EagerTensorProtocol


@typing.runtime_checkable
class EagerBackendAdapter(typing.Protocol):
    """Explicitly bound eager backend adapter for concrete operation, dtype, and submodule lookup."""

    @property
    def backend_name(self) -> str:
        """Return canonical backend identifier.

        Returns:
            str: Canonical backend name.
        """
        ...

    def bound_operations(self) -> Mapping[str, EagerBoundCallable]:
        """Return explicitly bound operation callables for this backend.

        Returns:
            Mapping[str, EagerBoundCallable]: Mapping from operation name to concrete callable.
        """
        ...

    def bound_dtypes(self) -> Mapping[str, BackendArg]:
        """Return explicitly bound dtype objects for this backend.

        Returns:
            Mapping[str, BackendArg]: Mapping from dtype name to concrete backend dtype.
        """
        ...

    def bound_submodules(self) -> Mapping[str, BackendModuleProtocol | types.ModuleType]:
        """Return explicitly bound submodules (e.g. `linalg`, `fft`, `random`) for this backend.

        Returns:
            Mapping[str, BackendModuleProtocol | types.ModuleType]: Mapping from submodule name to namespace.
        """
        ...

    def get_op(self, name: str) -> EagerBoundCallable | None:
        """Return bound operation `name`, or `None` if unsupported.

        Args:
            name (str): Operation identifier.

        Returns:
            EagerBoundCallable | None: Bound callable or `None`.
        """
        ...

    def require_op(self, name: str) -> EagerBoundCallable:
        """Return bound operation `name`, raising `AttributeError` if unsupported.

        Args:
            name (str): Operation identifier.

        Returns:
            EagerBoundCallable: Bound callable.
        """
        ...

    def has_op(self, name: str) -> bool:
        """Return True if operation `name` is supported by this adapter.

        Args:
            name (str): Operation identifier.

        Returns:
            bool: True when callable `name` is bound.
        """
        ...

    def get_dtype(self, name: str, default: BackendArg = None) -> BackendArg:
        """Return bound dtype `name` or `default` if absent.

        Args:
            name (str): Canonical dtype identifier (e.g. `'float32'`).
            default (BackendArg): Fallback value when absent.

        Returns:
            BackendArg: Bound dtype object or `default`.
        """
        ...

    def has_dtype(self, name: str) -> bool:
        """Return True if dtype `name` is bound on this adapter.

        Args:
            name (str): Canonical dtype identifier.

        Returns:
            bool: True when dtype `name` is available.
        """
        ...

    def get_submodule(self, name: str) -> BackendModuleProtocol | types.ModuleType | None:
        """Return bound submodule `name` (such as `'linalg'` or `'fft'`), or `None`.

        Args:
            name (str): Submodule identifier.

        Returns:
            BackendModuleProtocol | types.ModuleType | None: Bound submodule namespace or `None`.
        """
        ...

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
        """Resolve a single or dotted symbol path against bound submodules, operations, and dtypes.

        Args:
            dotted_path (str): Symbol name or dotted path (e.g. `'linalg.svd'`).
            default (BackendOperand): Fallback value if unresolved.

        Returns:
            tuple[bool, BackendOperand | types.ModuleType]: `(True, symbol)` if resolved, else `(False, default)`.
        """
        ...

    def has_symbol(self, dotted_path: str) -> bool:
        """Return True if `dotted_path` resolves on this adapter.

        Args:
            dotted_path (str): Symbol name or dotted path.

        Returns:
            bool: True when `dotted_path` is bound or structurally present.
        """
        ...

    def require_symbol(self, dotted_path: str) -> EagerTensorProtocol:
        """Resolve `dotted_path` on this adapter, raising `AttributeError` if absent.

        Args:
            dotted_path (str): Symbol name or dotted path.

        Returns:
            EagerTensorProtocol: Resolved symbol value.
        """
        ...

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
        default: BackendOperand | _MissingBackendAttrSentinel = ...,
    ) -> BackendOperand | types.ModuleType:
        """Return resolved symbol `dotted_path` or `default` when absent.

        Args:
            dotted_path (str): Symbol name or dotted path.
            default (BackendOperand | _MissingBackendAttrSentinel): Optional fallback value.

        Returns:
            BackendOperand | types.ModuleType: Resolved symbol or `default`.
        """
        ...

    def asarray(self, data: BackendArg, dtype: BackendArg = None) -> EagerTensorProtocol:
        """Convert `data` into a backend tensor via bound `asarray` / `array`.

        Args:
            data (BackendArg): Input array or scalar.
            dtype (BackendArg): Optional target dtype.

        Returns:
            EagerTensorProtocol: Backend tensor.
        """
        ...

    def zeros(self, shape: tuple[int, ...], dtype: BackendArg = None) -> EagerTensorProtocol:
        """Create a zero-filled backend tensor of `shape`.

        Args:
            shape (tuple[int, ...]): Target dimensions.
            dtype (BackendArg): Optional target dtype.

        Returns:
            EagerTensorProtocol: Zero-initialized backend tensor.
        """
        ...

    def ones(self, shape: tuple[int, ...], dtype: BackendArg = None) -> EagerTensorProtocol:
        """Create a one-filled backend tensor of `shape`.

        Args:
            shape (tuple[int, ...]): Target dimensions.
            dtype (BackendArg): Optional target dtype.

        Returns:
            EagerTensorProtocol: One-initialized backend tensor.
        """
        ...


_BoundOpCandidate = typing.Union[  # noqa: UP007
    CallableLike[BackendOperand],
    EagerBoundCallable,
    EagerCallableProtocol,
    CallableObjectProtocol,
    VariadicCallableProtocol,
    BackendOperand,
]


class StaticEagerBackendAdapter:
    """Concrete `EagerBackendAdapter` backed by explicit function, dtype, and submodule dicts."""

    def __init__(
        self,
        backend_name: str,
        ops: Mapping[str, _BoundOpCandidate] | None = None,
        dtypes: Mapping[str, BackendArg] | None = None,
        submodules: Mapping[str, BackendModuleProtocol | types.ModuleType] | None = None,
        module: types.ModuleType | BackendOperand | None = None,
    ) -> None:
        """Initialize static backend adapter bindings.

        Args:
            backend_name (str): Canonical backend name.
            ops (Mapping[str, _BoundOpCandidate] | None): Explicitly bound operation callables.
            dtypes (Mapping[str, BackendArg] | None): Explicitly bound backend dtype objects.
            submodules (Mapping[str, BackendModuleProtocol | types.ModuleType] | None): Explicitly bound submodules.
            module (types.ModuleType | BackendOperand | None): Optional live backend module namespace.
        """
        self._backend_name = backend_name
        self._ops: dict[str, EagerBoundCallable] = (
            {k: require_callable(v) for k, v in ops.items()} if ops is not None else {}
        )
        self._dtypes: dict[str, BackendArg] = dict(dtypes) if dtypes is not None else {}
        self._submodules: dict[str, BackendModuleProtocol | types.ModuleType] = (
            dict(submodules) if submodules is not None else {}
        )
        self._module = module

    @property
    def backend_name(self) -> str:
        """Return canonical backend identifier.

        Returns:
            str: Canonical backend name.
        """
        return self._backend_name

    @property
    def __name__(self) -> str:
        """Return wrapped module or generator class `__name__` when available.

        Returns:
            str: Module/class name or `backend_name` fallback.
        """
        if isinstance(self._module, (types.ModuleType, type)):
            return self._module.__name__
        return self._backend_name

    def bound_operations(self) -> Mapping[str, EagerBoundCallable]:
        """Return explicitly bound operation callables.

        Returns:
            Mapping[str, EagerBoundCallable]: Copy of bound operations mapping.
        """
        return dict(self._ops)

    def bound_dtypes(self) -> Mapping[str, BackendArg]:
        """Return explicitly bound dtype objects.

        Returns:
            Mapping[str, BackendArg]: Copy of bound dtypes mapping.
        """
        return dict(self._dtypes)

    def bound_submodules(self) -> Mapping[str, BackendModuleProtocol | types.ModuleType]:
        """Return explicitly bound submodules.

        Returns:
            Mapping[str, BackendModuleProtocol | types.ModuleType]: Copy of bound submodules mapping.
        """
        return dict(self._submodules)

    def _lookup_live(self, name: str) -> tuple[bool, EagerTensorProtocol | None]:
        """Check live module for `name` using static `@runtime_checkable` readers.

        Args:
            name (str): Member identifier.

        Returns:
            tuple[bool, EagerTensorProtocol | None]: `(True, value)` if present on live module, else `(False, None)`.
        """
        return read_static_attr(self._module, name) if self._module is not None else (False, None)

    def get_op(self, name: str) -> EagerBoundCallable | None:
        """Return bound operation `name`, checking live module overrides then static bindings.

        Args:
            name (str): Operation identifier.

        Returns:
            EagerBoundCallable | None: Bound callable or `None`.
        """
        found, live_val = self._lookup_live(name)
        if found and live_val is not None and callable(live_val):
            return require_callable(live_val)
        return self._ops.get(name)

    def require_op(self, name: str) -> EagerBoundCallable:
        """Return bound operation `name`, raising `AttributeError` when unavailable.

        Args:
            name (str): Operation identifier.

        Returns:
            EagerBoundCallable: Bound callable.

        Raises:
            AttributeError: If `name` is not bound on this backend adapter.
        """
        fn = self.get_op(name)
        if fn is None:
            raise AttributeError(f"Backend '{self._backend_name}' has no bound operation '{name}'")
        return fn

    def has_op(self, name: str) -> bool:
        """Return True if operation `name` is bound and callable.

        Args:
            name (str): Operation identifier.

        Returns:
            bool: True when operation `name` is available.
        """
        return self.get_op(name) is not None

    def get_dtype(self, name: str, default: BackendArg = None) -> BackendArg:
        """Return bound dtype `name` or `default` when unavailable.

        Args:
            name (str): Canonical dtype identifier.
            default (BackendArg): Fallback dtype value.

        Returns:
            BackendArg: Bound dtype object or `default`.
        """
        if name in self._dtypes:
            return self._dtypes[name]
        found, live_val = self._lookup_live(name)
        return live_val if (found and live_val is not None) else default

    def has_dtype(self, name: str) -> bool:
        """Return True if dtype `name` is bound.

        Args:
            name (str): Canonical dtype identifier.

        Returns:
            bool: True when dtype `name` is available.
        """
        if name in self._dtypes:
            return True
        found, live_val = self._lookup_live(name)
        return found and live_val is not None

    def get_submodule(self, name: str) -> BackendModuleProtocol | types.ModuleType | None:
        """Return bound submodule `name` or `None` when absent.

        Args:
            name (str): Submodule identifier.

        Returns:
            BackendModuleProtocol | types.ModuleType | None: Submodule namespace or `None`.
        """
        if name in self._submodules:
            return self._submodules[name]
        found, live_val = self._lookup_live(name)
        return live_val if (found and live_val is not None) else None

    def _resolve_single_key(
        self,
        key: str,
        default: BackendOperand,
    ) -> tuple[bool, BackendOperand | types.ModuleType]:
        """Resolve a single (non-dotted) symbol key against bound tables and live module.

        Args:
            key (str): Single symbol identifier.
            default (BackendOperand): Fallback value when unresolved.

        Returns:
            tuple[bool, BackendOperand | types.ModuleType]: `(True, resolved)` if found, else `(False, default)`.
        """
        if key in self._ops:
            return True, self._ops[key]
        if key in self._dtypes:
            return True, self._dtypes[key]
        if key in self._submodules:
            return True, self._submodules[key]
        found, live_val = self._lookup_live(key)
        return (True, live_val) if found else (False, default)

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
        """Resolve a single or dotted symbol path against bound submodules, operations, and dtypes.

        Args:
            dotted_path (str): Symbol name or dotted path.
            default (BackendOperand): Fallback value when unresolved.

        Returns:
            tuple[bool, BackendOperand | types.ModuleType]: `(True, resolved)` if found, else `(False, default)`.
        """
        parts = [p for p in dotted_path.split(".") if p]
        if not parts:
            return False, default
        if len(parts) == 1:
            return self._resolve_single_key(parts[0], default)
        curr: types.ModuleType | BackendOperand = (
            self._submodules[parts[0]] if parts[0] in self._submodules else self._module
        )
        start_idx = 1 if parts[0] in self._submodules else 0
        for part in parts[start_idx:]:
            if curr is None:
                return False, default
            found, next_val = read_static_attr(curr, part)
            if not found:
                return False, default
            curr = next_val
        return True, curr

    def has_symbol(self, dotted_path: str) -> bool:
        """Return True if `dotted_path` resolves on this adapter.

        Args:
            dotted_path (str): Symbol name or dotted path.

        Returns:
            bool: True when `dotted_path` resolves.
        """
        return self.resolve_symbol(dotted_path)[0]

    def _require_single_live_symbol(self, key: str) -> EagerTensorProtocol:
        """Resolve single symbol `key` as `EagerTensorProtocol` or raise `AttributeError`.

        Args:
            key (str): Single symbol identifier.

        Returns:
            EagerTensorProtocol: Resolved live symbol or bound module protocol.

        Raises:
            AttributeError: If `key` cannot be resolved to a non-None symbol.
        """
        found, live_val = self._lookup_live(key)
        if found and live_val is not None:
            return live_val
        if key in self._submodules:
            sub = self._submodules[key]
            if not isinstance(sub, types.ModuleType):
                return sub
        raise AttributeError(key)

    def require_symbol(self, dotted_path: str) -> EagerTensorProtocol:
        """Resolve `dotted_path` on this adapter, raising `AttributeError` if absent.

        Args:
            dotted_path (str): Symbol name or dotted path.

        Returns:
            EagerTensorProtocol: Resolved symbol value.

        Raises:
            AttributeError: If `dotted_path` cannot be resolved.
        """
        parts = [p for p in dotted_path.split(".") if p]
        if not parts:
            raise AttributeError(dotted_path)
        if len(parts) == 1:
            return self._require_single_live_symbol(parts[0])
        curr: types.ModuleType | BackendOperand = (
            self._submodules[parts[0]] if parts[0] in self._submodules else self._module
        )
        start_idx = 1 if parts[0] in self._submodules else 0
        last_val: EagerTensorProtocol | None = None
        for part in parts[start_idx:]:
            if curr is None:
                raise AttributeError(dotted_path)
            found, last_val = read_static_attr(curr, part)
            if not found:
                raise AttributeError(dotted_path)
            curr = last_val
        if last_val is None:
            raise AttributeError(dotted_path)
        return last_val

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
        """Return resolved symbol `dotted_path` or `default` when absent.

        Args:
            dotted_path (str): Symbol name or dotted path.
            default (BackendOperand | _MissingBackendAttrSentinel): Optional fallback value.

        Returns:
            BackendOperand | types.ModuleType: Resolved symbol or `default`.

        Raises:
            AttributeError: If `dotted_path` is absent and no `default` was supplied.
        """
        default_arg: BackendOperand = None if isinstance(default, _MissingBackendAttrSentinel) else default
        found, val = self.resolve_symbol(dotted_path, default_arg)
        if not found and isinstance(default, _MissingBackendAttrSentinel):
            raise AttributeError(dotted_path)
        return val

    def asarray(self, data: BackendArg, dtype: BackendArg = None) -> EagerTensorProtocol:
        """Convert `data` into a backend tensor via bound `asarray` or `array`.

        Args:
            data (BackendArg): Input array or scalar.
            dtype (BackendArg): Optional target dtype.

        Returns:
            EagerTensorProtocol: Converted backend tensor.
        """
        fn = self.get_op("asarray") or self.require_op("array")
        return fn(data) if dtype is None else fn(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: BackendArg = None) -> EagerTensorProtocol:
        """Create a zero-filled backend tensor of `shape`.

        Args:
            shape (tuple[int, ...]): Target dimensions.
            dtype (BackendArg): Optional target dtype.

        Returns:
            EagerTensorProtocol: Zero-initialized backend tensor.
        """
        fn = self.require_op("zeros")
        return fn(shape) if dtype is None else fn(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: BackendArg = None) -> EagerTensorProtocol:
        """Create a one-filled backend tensor of `shape`.

        Args:
            shape (tuple[int, ...]): Target dimensions.
            dtype (BackendArg): Optional target dtype.

        Returns:
            EagerTensorProtocol: One-initialized backend tensor.
        """
        fn = self.require_op("ones")
        return fn(shape) if dtype is None else fn(shape, dtype=dtype)


class ModuleEagerBackendAdapter(EagerBackendAdapter, typing.Protocol):
    """Specialized `EagerBackendAdapter` for `BackendModuleProtocol` targets narrowing `None` defaults."""

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
    def get_symbol(self, dotted_path: str, default: None) -> EagerTensorProtocol | None: ...

    @typing.overload
    def get_symbol(self, dotted_path: str, default: BackendOperand) -> EagerTensorProtocol | BackendArg: ...

    def get_symbol(
        self,
        dotted_path: str,
        default: BackendOperand | _MissingBackendAttrSentinel = ...,
    ) -> BackendOperand | types.ModuleType:
        """Return resolved module symbol `dotted_path` or `default` when absent.

        Args:
            dotted_path (str): Symbol name or dotted path.
            default (BackendOperand | _MissingBackendAttrSentinel): Optional fallback value.

        Returns:
            BackendOperand | types.ModuleType: Resolved symbol or `default`.
        """
        ...


@typing.overload
def adapt_backend_module(
    target: BackendModuleProtocol | BackendNamespaceProtocol | types.ModuleType | None,
    backend_name: str = ...,
) -> ModuleEagerBackendAdapter: ...


@typing.overload
def adapt_backend_module(
    target: EagerBackendAdapter | BackendOperand,
    backend_name: str = ...,
) -> EagerBackendAdapter: ...


def adapt_backend_module(
    target: types.ModuleType | EagerBackendAdapter | BackendOperand,
    backend_name: str = "dynamic",
) -> EagerBackendAdapter:
    """Return `target` if already an `EagerBackendAdapter`, else wrap it in `StaticEagerBackendAdapter`.

    Args:
        target (types.ModuleType | EagerBackendAdapter | BackendOperand): Module or existing adapter.
        backend_name (str): Canonical backend identifier when wrapping a module.

    Returns:
        EagerBackendAdapter: Concrete backend adapter instance.
    """
    if isinstance(target, StaticEagerBackendAdapter):
        return target
    if not isinstance(target, types.ModuleType) and isinstance(target, EagerBackendAdapter):
        if type(target).__module__.startswith("ml_switcheroo_compiler"):
            return target
        return StaticEagerBackendAdapter(backend_name=backend_name, module=target)
    return StaticEagerBackendAdapter(backend_name=backend_name, module=target)


__all__ = [
    "BackendGraphProtocol",
    "BackendModuleProtocol",
    "BackendNamespaceProtocol",
    "BackendNodeProtocol",
    "BackendTensorProtocol",
    "EagerBackendAdapter",
    "EagerCallableProtocol",
    "BackendOperand",
    "EagerTensorProtocol",
    "BackendArg",
    "ShapeProtocol",
    "StaticEagerBackendAdapter",
    "adapt_backend_module",
    "coerce_float",
    "coerce_int",
    "coerce_shape",
]
