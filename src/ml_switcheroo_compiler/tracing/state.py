"""Active graph builder tracing context (State Layer 3).

Manages the dynamic graph stack, stream scopes, loop invariants, and source AST
linkage while recording ``LogicalNode`` instances into an active ``LogicalGraph``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import threading
import typing
import weakref
from collections.abc import Iterator, Sequence
from contextvars import ContextVar

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.linker import get_source_ast_ref

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir import AttributeValue, TensorSpec

    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.autograph import LoopOptions, ShapeInvariantType


@typing.runtime_checkable
class _GraphInputsProtocol(typing.Protocol):
    """Protocol for graph objects exposing `inputs`."""

    inputs: list[str]


@typing.runtime_checkable
class _GraphInputSpecsProtocol(typing.Protocol):
    """Protocol for graph objects exposing `input_specs`."""

    input_specs: dict[str, TensorSpec]


@typing.runtime_checkable
class _NodeAstRefProtocol(typing.Protocol):
    """Protocol for nodes exposing `source_ast_ref`."""

    source_ast_ref: str | None


@typing.runtime_checkable
class _NodeDomainProtocol(typing.Protocol):
    """Protocol for nodes exposing `domain`."""

    domain: str


@typing.runtime_checkable
class _NodeStreamProtocol(typing.Protocol):
    """Protocol for nodes exposing `stream`."""

    stream: str | None


@typing.runtime_checkable
class _NodeLoopAttrsProtocol(typing.Protocol):
    """Protocol for loop nodes exposing `op_type` and `attributes`."""

    op_type: str
    attributes: dict[str, AttributeValue]


@typing.runtime_checkable
class _NodeOpTypeProtocol(typing.Protocol):
    """Protocol for nodes exposing `id` and `op_type`."""

    id: str
    op_type: str


class _NodeIdOnlyProtocol(typing.Protocol):
    """Protocol for minimal node stubs exposing only `id`."""

    id: str


class _GraphNameProtocol(typing.Protocol):
    """Protocol for graphs exposing `name`."""

    name: str | None


def _has_graph_inputs(val: LogicalGraph | _GraphNameProtocol) -> typing.TypeGuard[_GraphInputsProtocol]:
    """Check whether `val` exposes `inputs`.

    Args:
        val (LogicalGraph | _GraphNameProtocol): Active graph.

    Returns:
        bool: True if `val` has `inputs`.
    """
    return isinstance(val, _GraphInputsProtocol)


def _has_graph_input_specs(val: LogicalGraph | _GraphInputsProtocol) -> typing.TypeGuard[_GraphInputSpecsProtocol]:
    """Check whether `val` exposes `input_specs`.

    Args:
        val (LogicalGraph | _GraphInputsProtocol): Active graph.

    Returns:
        bool: True if `val` has `input_specs`.
    """
    return isinstance(val, _GraphInputSpecsProtocol)


def _has_node_ast_ref(val: LogicalNode | _NodeIdOnlyProtocol) -> typing.TypeGuard[_NodeAstRefProtocol]:
    """Check whether `val` exposes `source_ast_ref`.

    Args:
        val (LogicalNode | _NodeIdOnlyProtocol): Target node.

    Returns:
        bool: True if `val` has `source_ast_ref`.
    """
    return isinstance(val, _NodeAstRefProtocol)


def _has_node_domain(val: LogicalNode | _NodeIdOnlyProtocol) -> typing.TypeGuard[_NodeDomainProtocol]:
    """Check whether `val` exposes `domain`.

    Args:
        val (LogicalNode | _NodeIdOnlyProtocol): Target node.

    Returns:
        bool: True if `val` has `domain`.
    """
    return isinstance(val, _NodeDomainProtocol)


def _has_node_stream(val: LogicalNode | _NodeIdOnlyProtocol) -> typing.TypeGuard[_NodeStreamProtocol]:
    """Check whether `val` exposes `stream`.

    Args:
        val (LogicalNode | _NodeIdOnlyProtocol): Target node.

    Returns:
        bool: True if `val` has `stream`.
    """
    return isinstance(val, _NodeStreamProtocol)


def _has_node_loop_attrs(val: LogicalNode | _NodeIdOnlyProtocol) -> typing.TypeGuard[_NodeLoopAttrsProtocol]:
    """Check whether `val` exposes `op_type` and `attributes`.

    Args:
        val (LogicalNode | _NodeIdOnlyProtocol): Target node.

    Returns:
        bool: True if `val` has `op_type` and `attributes`.
    """
    return isinstance(val, _NodeLoopAttrsProtocol)


def _has_node_op_type(val: LogicalNode | _NodeIdOnlyProtocol) -> typing.TypeGuard[_NodeOpTypeProtocol]:
    """Check whether `val` exposes `id` and `op_type`.

    Args:
        val (LogicalNode | _NodeIdOnlyProtocol): Target node.

    Returns:
        bool: True if `val` has `id` and `op_type`.
    """
    return isinstance(val, _NodeOpTypeProtocol)


_LIVE_TENSORS: dict[int, weakref.ref[Tensor]] = {}
_TENSOR_REGISTRY_LOCK = threading.Lock()
_SCOPED_LIVE_TENSORS: ContextVar[dict[int, weakref.ref[Tensor]] | None] = ContextVar(
    "scoped_live_tensors",
    default=None,
)


def _bind_weak_tensor(table: dict[int, weakref.ref[Tensor]], tensor: Tensor) -> None:
    """Insert a weak reference to `tensor` into `table` with an O(1) eviction callback.

    Args:
        table (dict[int, weakref.ref[Tensor]]): Target identity-keyed weak reference table.
        tensor (Tensor): Live tensor instance to register weakly.
    """
    tid = id(tensor)

    def _evict(_dead_ref: weakref.ref[Tensor]) -> None:
        """Remove `tid` entry from `table` once `tensor` is garbage-collected.

        Args:
            _dead_ref (weakref.ref[Tensor]): Expired weak reference.
        """
        table.pop(tid, None)

    table[tid] = weakref.ref(tensor, _evict)


def register_live_tensor(tensor: Tensor) -> None:
    """Register `tensor` weakly so imperative autodiff resolves graph inputs in O(1) without GC heap scans.

    Args:
        tensor (Tensor): Live tensor instance to track weakly.
    """
    scoped = _SCOPED_LIVE_TENSORS.get()
    if scoped is not None:
        _bind_weak_tensor(scoped, tensor)
    with _TENSOR_REGISTRY_LOCK:
        _bind_weak_tensor(_LIVE_TENSORS, tensor)


def get_live_tensors() -> list[Tensor]:
    """Return all currently live `Tensor` instances in the active weak registry scope.

    Returns:
        list[Tensor]: Snapshot list of strongly reachable `Tensor` instances.
    """
    scoped = _SCOPED_LIVE_TENSORS.get()
    if scoped is not None:
        refs = list(scoped.values())
    else:
        with _TENSOR_REGISTRY_LOCK:
            refs = list(_LIVE_TENSORS.values())
    return [obj for ref in refs for obj in (ref(),) if obj is not None]


@contextlib.contextmanager
def live_tensor_scope(initial: Sequence[Tensor] = ()) -> Iterator[dict[int, weakref.ref[Tensor]]]:
    """Create an isolated `ContextVar` weak-reference scope for deterministic live-tensor resolution.

    Args:
        initial (Sequence[Tensor]): Optional initial tensors to populate into the scope.

    Yields:
        dict[int, weakref.ref[Tensor]]: Isolated identity-keyed weak table active within the `with` block.
    """
    scoped_table: dict[int, weakref.ref[Tensor]] = {}
    for item in initial:
        _bind_weak_tensor(scoped_table, item)
    token = _SCOPED_LIVE_TENSORS.set(scoped_table)
    try:
        yield scoped_table
    finally:
        _SCOPED_LIVE_TENSORS.reset(token)


@dataclasses.dataclass
class _TracingContextFrame:
    """Per-context tracing frame stored in a `ContextVar` for thread and async-task isolation."""

    is_tracing: bool = False
    active_graph: LogicalGraph | None = None
    graph_stack: list[LogicalGraph] = dataclasses.field(default_factory=list)
    constant_cache: dict[int, str] = dataclasses.field(default_factory=dict)
    current_loop_options: LoopOptions | None = None
    loop_options_stack: list[LoopOptions | None] = dataclasses.field(default_factory=list)

    def clone(self) -> _TracingContextFrame:
        """Create a shallow copy-on-write clone of this context frame.

        Returns:
            _TracingContextFrame: Cloned context frame with independent stack and cache containers.
        """
        return _TracingContextFrame(
            is_tracing=self.is_tracing,
            active_graph=self.active_graph,
            graph_stack=list(self.graph_stack),
            constant_cache=dict(self.constant_cache),
            current_loop_options=self.current_loop_options,
            loop_options_stack=list(self.loop_options_stack),
        )


class TracingState:
    """ContextVar-isolated state manager for thread-safe and async-task-safe graph tracing."""

    def __init__(self) -> None:
        """Initialize an isolated `ContextVar` backing store for this `TracingState` instance."""
        self._ctx_var: ContextVar[_TracingContextFrame | None] = ContextVar(
            f"tracing_state_{id(self)}",
            default=None,
        )

    def _frame(self) -> _TracingContextFrame:
        """Return the active context frame, lazily initializing per thread or async context.

        Returns:
            _TracingContextFrame: Current context frame.
        """
        frame = self._ctx_var.get()
        if frame is None:
            frame = _TracingContextFrame()
            self._ctx_var.set(frame)
        return frame

    @property
    def is_tracing(self) -> bool:
        """Return whether tracing is active in the current execution context.

        Returns:
            bool: True when an active trace is recording operations.
        """
        return self._frame().is_tracing

    @is_tracing.setter
    def is_tracing(self, value: bool) -> None:
        """Set `is_tracing` using copy-on-write `ContextVar` state.

        Args:
            value (bool): New tracing status flag.
        """
        nxt = self._frame().clone()
        nxt.is_tracing = value
        self._ctx_var.set(nxt)

    @is_tracing.deleter
    def is_tracing(self) -> None:
        """Reset `is_tracing` to `False` when restored via `unittest.mock.patch` cleanup."""
        self.is_tracing = False

    @property
    def active_graph(self) -> LogicalGraph | None:
        """Return the active `LogicalGraph` in the current execution context.

        Returns:
            LogicalGraph | None: Active graph or `None`.
        """
        return self._frame().active_graph

    @active_graph.setter
    def active_graph(self, value: LogicalGraph | None) -> None:
        """Set `active_graph` using copy-on-write `ContextVar` state.

        Args:
            value (LogicalGraph | None): Graph to make active, or `None`.
        """
        nxt = self._frame().clone()
        nxt.active_graph = value
        self._ctx_var.set(nxt)

    @active_graph.deleter
    def active_graph(self) -> None:
        """Reset `active_graph` to `None` when restored via `unittest.mock.patch` cleanup."""
        self.active_graph = None

    @property
    def graph_stack(self) -> list[LogicalGraph]:
        """Return the nested graph stack for the current execution context.

        Returns:
            list[LogicalGraph]: Enclosing active graphs.
        """
        return self._frame().graph_stack

    @graph_stack.setter
    def graph_stack(self, value: list[LogicalGraph]) -> None:
        """Replace `graph_stack` using copy-on-write `ContextVar` state.

        Args:
            value (list[LogicalGraph]): Replacement graph stack.
        """
        nxt = self._frame().clone()
        nxt.graph_stack = list(value)
        self._ctx_var.set(nxt)

    @graph_stack.deleter
    def graph_stack(self) -> None:
        """Reset `graph_stack` to empty list when restored via `unittest.mock.patch` cleanup."""
        self.graph_stack = []

    @property
    def constant_cache(self) -> dict[int, str]:
        """Return the constant deduplication cache for the current execution context.

        Returns:
            dict[int, str]: Mapping from object identity to constant node ID.
        """
        return self._frame().constant_cache

    @constant_cache.setter
    def constant_cache(self, value: dict[int, str]) -> None:
        """Replace `constant_cache` using copy-on-write `ContextVar` state.

        Args:
            value (dict[int, str]): Replacement constant cache mapping.
        """
        nxt = self._frame().clone()
        nxt.constant_cache = dict(value)
        self._ctx_var.set(nxt)

    @constant_cache.deleter
    def constant_cache(self) -> None:
        """Reset `constant_cache` to empty dict when restored via `unittest.mock.patch` cleanup."""
        self.constant_cache = {}

    @property
    def current_loop_options(self) -> LoopOptions | None:
        """Return active loop metadata options in the current execution context.

        Returns:
            LoopOptions | None: Current loop options or `None`.
        """
        return self._frame().current_loop_options

    @current_loop_options.setter
    def current_loop_options(self, value: LoopOptions | None) -> None:
        """Set `current_loop_options` using copy-on-write `ContextVar` state.

        Args:
            value (LoopOptions | None): Loop options to activate, or `None`.
        """
        nxt = self._frame().clone()
        nxt.current_loop_options = value
        self._ctx_var.set(nxt)

    @current_loop_options.deleter
    def current_loop_options(self) -> None:
        """Reset `current_loop_options` to `None` when restored via `unittest.mock.patch` cleanup."""
        self.current_loop_options = None

    @property
    def loop_options_stack(self) -> list[LoopOptions | None]:
        """Return the nested loop options stack for the current execution context.

        Returns:
            list[LoopOptions | None]: Saved loop options stack.
        """
        return self._frame().loop_options_stack

    @loop_options_stack.setter
    def loop_options_stack(self, value: list[LoopOptions | None]) -> None:
        """Replace `loop_options_stack` using copy-on-write `ContextVar` state.

        Args:
            value (list[LoopOptions | None]): Replacement loop options stack.
        """
        nxt = self._frame().clone()
        nxt.loop_options_stack = list(value)
        self._ctx_var.set(nxt)

    @loop_options_stack.deleter
    def loop_options_stack(self) -> None:
        """Reset `loop_options_stack` to empty list when restored via `unittest.mock.patch` cleanup."""
        self.loop_options_stack = []

    def _enrich_ast_and_domain(self, node: LogicalNode) -> None:
        """Enrich a node with AST and domain information.

        Args:
            node (LogicalNode): The IR node to enrich.
        """
        if _has_node_ast_ref(node) and node.source_ast_ref is None:
            node.source_ast_ref = get_source_ast_ref()
        active_g = self.active_graph
        if active_g is not None and active_g.name and _has_node_domain(node) and node.domain in (None, ""):
            node.domain = active_g.name

    def _enrich_stream(self, node: LogicalNode) -> None:
        """Enrich a node with current stream information.

        Args:
            node (LogicalNode): The IR node to enrich.
        """
        if "ml_switcheroo_compiler.core.config" not in sys.modules:
            return
        from ml_switcheroo_compiler.core.config import config as active_config

        if _has_node_stream(node) and node.stream is None and active_config.current_stream != "default":
            node.stream = active_config.current_stream

    def _enrich_node(self, node: LogicalNode) -> None:
        """Enrich a newly created node with implicit context metadata (AST, domain, stream).

        Args:
            node (LogicalNode): The IR node to enrich with metadata.
        """
        self._enrich_node_fields(node)

    def _enrich_node_fields(self, node: LogicalNode) -> None:
        """Populate AST, domain, and stream fields on `node`.

        Args:
            node (LogicalNode): The IR node to enrich with metadata.
        """
        self._enrich_ast_and_domain(node)
        self._enrich_stream(node)

    def _apply_loop_options_to_node(self, node: LogicalNode) -> None:
        """Attach active loop options metadata to a Loop or WhileLoop node.

        Args:
            node (LogicalNode): The IR node to populate with loop options.
        """
        if (
            self.current_loop_options is None
            or not _has_node_loop_attrs(node)
            or node.op_type not in ("Loop", "WhileLoop")
            or not isinstance(node.attributes, dict)
        ):
            return
        attrs = node.attributes
        opts = self.current_loop_options
        opt_items: tuple[tuple[str, int | bool | ShapeInvariantType | None], ...] = (
            ("parallel_iterations", opts.parallel_iterations),
            ("swap_memory", opts.swap_memory),
            ("maximum_iterations", opts.maximum_iterations),
            ("shape_invariants", opts.shape_invariants),
        )
        for key, val in opt_items:
            if key not in attrs and val is not None:
                attrs[key] = val
        if "loop_options" not in attrs:
            attrs["loop_options"] = opts

    def add_node(self, node: LogicalNode) -> None:
        """Register a node into the currently active trace graph.

        Args:
            node (LogicalNode): The IR node to append to the computational graph.
        """
        if not self.is_tracing or self.active_graph is None:
            return

        self._enrich_node(node)
        self._apply_loop_options_to_node(node)

        self.active_graph.nodes[node.id] = node
        op_type = node.op_type if _has_node_op_type(node) else None
        if op_type != "Input" and _has_graph_inputs(self.active_graph):
            if node.id in self.active_graph.inputs:
                self.active_graph.inputs.remove(node.id)
                if _has_graph_input_specs(self.active_graph):
                    self.active_graph.input_specs.pop(node.id, None)

    def start_tracing(self, name: str = "Model") -> LogicalGraph:
        """Activate the tracing context and initialize a new empty graph.

        Args:
            name (str): The logical name assigned to the computational graph.

        Returns:
            LogicalGraph: The newly initialized LogicalGraph instance.
        """
        nxt = self._frame().clone()
        if nxt.active_graph is not None:
            nxt.graph_stack.append(nxt.active_graph)
            nxt.loop_options_stack.append(nxt.current_loop_options)
        new_graph = LogicalGraph(name=name)
        nxt.active_graph = new_graph
        nxt.constant_cache = {}
        nxt.current_loop_options = None
        nxt.is_tracing = True
        self._ctx_var.set(nxt)
        return new_graph

    def stop_tracing(self) -> LogicalGraph | None:
        """Deactivate the tracing context and return the captured graph.

        Returns:
            LogicalGraph | None: The populated LogicalGraph containing all operations captured during tracing.
        """
        nxt = self._frame().clone()
        graph = nxt.active_graph
        if nxt.graph_stack:
            nxt.active_graph = nxt.graph_stack.pop()
            nxt.current_loop_options = nxt.loop_options_stack.pop() if nxt.loop_options_stack else None
            nxt.is_tracing = True
        else:
            nxt.active_graph = None
            nxt.current_loop_options = None
            nxt.loop_options_stack.clear()
            nxt.is_tracing = False
        self._ctx_var.set(nxt)
        return graph


global_tracing_state = TracingState()
