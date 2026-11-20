"""JIT compilation pipeline, `disable_jit` context manager, and symbolic `eval_shape`."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import threading
from collections.abc import Iterator
from contextvars import ContextVar

import pydantic
from ml_switcheroo_ir import LogicalGraph, LogicalNode, TensorSpec

from ml_switcheroo_compiler.backends.registry import BackendRegistry, get_active_backend
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    IRAttrValue,
    OpValue,
    VariadicCallableProtocol,
    is_traced_data,
    is_variadic_callable,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, increment_trace_count
from ml_switcheroo_compiler.transforms import (
    PassManager,
    common_subexpression_elimination_pass,
    constant_folding_pass,
    dead_code_elimination_pass,
)
from ml_switcheroo_compiler.tree_util import TreeDef, tree_flatten, tree_unflatten

from .options import JitOptions as OptionsJitOptions
from .utils import DifferentiableFn, PrimalTree, as_primal_tree

_JIT_DISABLED_VAR: ContextVar[bool] = ContextVar("switcheroo_jit_disabled", default=False)
_JIT_CACHE_LOCK = threading.Lock()
_MAX_JIT_CACHE_ENTRIES: int = 512


class JitOptions(pydantic.BaseModel):
    """Configuration options for JIT compilation."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    static_argnums: IRAttrValue = None
    static_argnames: IRAttrValue = None
    donate_argnums: IRAttrValue = None
    donate_argnames: IRAttrValue = None
    keep_unused: bool = False
    device: IRAttrValue = None
    backend: IRAttrValue = None
    inline: bool = False
    abstracted_axes: IRAttrValue = None


@dataclasses.dataclass(frozen=True)
class _NormalizedJitConfig:
    """Canonicalized immutable configuration extracted from `JitOptions`."""

    static_argnums: frozenset[int]
    static_argnames: frozenset[str]
    keep_unused: bool
    backend: str | None
    device: str | None
    inline: bool


@dataclasses.dataclass(frozen=True)
class _CompiledOutputLeaf:
    """Descriptor for reconstructing a single PyTree leaf from a compiled JIT graph."""

    is_node: bool
    node_id: str
    tensor_config: TensorConfig | None
    literal_value: OpValue


@dataclasses.dataclass(frozen=True)
class _CompiledJitEntry:
    """Cached compiled `LogicalGraph` artifact and PyTree reconstruction metadata."""

    graph: LogicalGraph
    dyn_input_ids: tuple[str, ...]
    out_leaves: tuple[_CompiledOutputLeaf, ...]
    out_treedef: TreeDef
    backend_name: str
    device_override: str | None


_JIT_CACHE: collections.OrderedDict[str, _CompiledJitEntry] = collections.OrderedDict()


def is_jit_disabled() -> bool:
    """Return whether `disable_jit` is active in the current execution context.

    Returns:
        bool: True when JIT compilation is temporarily disabled.
    """
    return bool(_JIT_DISABLED_VAR.get())


@contextlib.contextmanager
def disable_jit() -> Iterator[None]:
    """Temporarily disable JIT compilation within the enclosed `ContextVar` scope.

    Yields:
        None: Execution scope where `jit`-wrapped functions execute eagerly.
    """
    token = _JIT_DISABLED_VAR.set(True)
    try:
        yield
    finally:
        _JIT_DISABLED_VAR.reset(token)


def clear_jit_cache() -> None:
    """Clear all compiled `LogicalGraph` entries from the in-memory JIT cache."""
    with _JIT_CACHE_LOCK:
        _JIT_CACHE.clear()


def get_jit_cache_size() -> int:
    """Return the number of compiled artifacts currently held in the JIT cache.

    Returns:
        int: Active JIT cache entry count.
    """
    with _JIT_CACHE_LOCK:
        return len(_JIT_CACHE)


def set_max_jit_cache_entries(limit: int) -> int:
    """Set the maximum number of compiled artifacts retained in `_JIT_CACHE` and return the prior limit.

    Args:
        limit (int): Positive maximum cache capacity.

    Returns:
        int: Previous cache capacity limit.

    Raises:
        ValueError: If `limit` is less than 1.
    """
    global _MAX_JIT_CACHE_ENTRIES
    if limit < 1:
        raise ValueError("limit must be >= 1")
    with _JIT_CACHE_LOCK:
        prev = _MAX_JIT_CACHE_ENTRIES
        _MAX_JIT_CACHE_ENTRIES = limit
        while len(_JIT_CACHE) > _MAX_JIT_CACHE_ENTRIES:
            _JIT_CACHE.popitem(last=False)
        return prev


def _parse_int_set(raw: IRAttrValue) -> frozenset[int]:
    """Normalize an `int` or sequence of `int`s into a `frozenset[int]`.

    Args:
        raw (IRAttrValue): Raw `static_argnums` option value.

    Returns:
        frozenset[int]: Normalized non-boolean integer indices.
    """
    if isinstance(raw, int) and not isinstance(raw, bool):
        return frozenset({raw})
    if isinstance(raw, (tuple, list)):
        return frozenset(v for v in raw if isinstance(v, int) and not isinstance(v, bool))
    return frozenset()


def _parse_str_set(raw: IRAttrValue) -> frozenset[str]:
    """Normalize a `str` or sequence of `str`s into a `frozenset[str]`.

    Args:
        raw (IRAttrValue): Raw `static_argnames` option value.

    Returns:
        frozenset[str]: Normalized argument names.
    """
    if isinstance(raw, str):
        return frozenset({raw})
    if isinstance(raw, (tuple, list)):
        return frozenset(v for v in raw if isinstance(v, str))
    return frozenset()


def _normalize_jit_options(options: JitOptions | OptionsJitOptions | None) -> _NormalizedJitConfig:
    """Convert either `grad.jit.JitOptions` or `grad.options.JitOptions` into `_NormalizedJitConfig`.

    Args:
        options (JitOptions | OptionsJitOptions | None): Caller-supplied JIT options.

    Returns:
        _NormalizedJitConfig: Canonicalized immutable JIT configuration.
    """
    if options is None:
        return _NormalizedJitConfig(frozenset(), frozenset(), False, None, None, False)
    if isinstance(options, OptionsJitOptions):
        return _NormalizedJitConfig(
            static_argnums=frozenset(options.static_argnums),
            static_argnames=frozenset(),
            keep_unused=False,
            backend=options.backend,
            device=options.device,
            inline=options.inline,
        )
    backend_str = str(options.backend) if options.backend is not None else None
    device_str = str(options.device) if options.device is not None else None
    return _NormalizedJitConfig(
        static_argnums=_parse_int_set(options.static_argnums),
        static_argnames=_parse_str_set(options.static_argnames),
        keep_unused=bool(options.keep_unused),
        backend=backend_str,
        device=device_str,
        inline=bool(options.inline),
    )


def _partition_call_args(
    args: tuple[OpValue, ...],
    kwargs: dict[str, OpValue],
    cfg: _NormalizedJitConfig,
) -> tuple[list[tuple[int, OpValue]], list[OpValue], dict[str, OpValue], dict[str, OpValue]]:
    """Split positional and keyword call arguments into static and dynamic subsets.

    Args:
        args (tuple[OpValue, ...]): Positional call arguments.
        kwargs (dict[str, OpValue]): Keyword call arguments.
        cfg (_NormalizedJitConfig): Normalized JIT configuration.

    Returns:
        tuple[list[tuple[int, OpValue]], list[OpValue], dict[str, OpValue], dict[str, OpValue]]:
            `(static_pos, dyn_pos, static_kw, dyn_kw)`.
    """
    static_pos: list[tuple[int, OpValue]] = []
    dyn_pos: list[OpValue] = []
    for idx, val in enumerate(args):
        if idx in cfg.static_argnums:
            static_pos.append((idx, val))
        else:
            dyn_pos.append(val)
    static_kw: dict[str, OpValue] = {}
    dyn_kw: dict[str, OpValue] = {}
    for key, val in kwargs.items():
        if key in cfg.static_argnames:
            static_kw[key] = val
        else:
            dyn_kw[key] = val
    return static_pos, dyn_pos, static_kw, dyn_kw


def _merge_call_args(
    total_pos: int,
    static_pos: list[tuple[int, OpValue]],
    dyn_pos: list[OpValue],
    static_kw: dict[str, OpValue],
    dyn_kw: dict[str, OpValue],
) -> tuple[list[OpValue], dict[str, OpValue]]:
    """Reassemble positional and keyword arguments from static and traced dynamic partitions.

    Args:
        total_pos (int): Total number of original positional arguments.
        static_pos (list[tuple[int, OpValue]]): Indexed static positional arguments.
        dyn_pos (list[OpValue]): Reconstructed dynamic positional arguments.
        static_kw (dict[str, OpValue]): Static keyword arguments.
        dyn_kw (dict[str, OpValue]): Reconstructed dynamic keyword arguments.

    Returns:
        tuple[list[OpValue], dict[str, OpValue]]: Full `(positional_args, keyword_args)` for `fun`.
    """
    static_map = dict(static_pos)
    dyn_iter = iter(dyn_pos)
    merged_pos: list[OpValue] = [static_map[i] if i in static_map else next(dyn_iter) for i in range(total_pos)]
    merged_kw = dict(dyn_kw)
    merged_kw.update(static_kw)
    return merged_pos, merged_kw


def _optimize_jit_graph(graph: LogicalGraph, keep_unused: bool) -> LogicalGraph:
    """Run middle-end `PassManager` optimizations and optional dead-input pruning on `graph`.

    Args:
        graph (LogicalGraph): Captured trace graph.
        keep_unused (bool): Whether to retain unreferenced `Input` nodes in `graph.inputs`.

    Returns:
        LogicalGraph: Optimized `LogicalGraph`.
    """
    pm = PassManager()
    pm.add_pass(constant_folding_pass)
    pm.add_pass(common_subexpression_elimination_pass)
    pm.add_pass(dead_code_elimination_pass)
    opt_graph = pm.run(graph)
    if not keep_unused:
        used_ids: set[str] = set(opt_graph.outputs)
        for node in opt_graph.nodes.values():
            used_ids.update(node.inputs)
        retained_inputs: list[str] = []
        for inp_id in opt_graph.inputs:
            if inp_id in used_ids:
                retained_inputs.append(inp_id)
            else:
                opt_graph.nodes.pop(inp_id, None)
        opt_graph.inputs = retained_inputs
    return opt_graph


def _trace_and_compile_jit(
    fun: DifferentiableFn,
    total_pos: int,
    static_pos: list[tuple[int, OpValue]],
    static_kw: dict[str, OpValue],
    dyn_leaves: list[OpValue],
    dyn_treedef: TreeDef,
    cfg: _NormalizedJitConfig,
    target_backend: str,
) -> _CompiledJitEntry:
    """Trace `fun` on symbolic `ProxyTensor` inputs, optimize the IR graph, and build a cache entry.

    Args:
        fun (DifferentiableFn): Target function to trace.
        total_pos (int): Total original positional argument count.
        static_pos (list[tuple[int, OpValue]]): Static positional arguments.
        static_kw (dict[str, OpValue]): Static keyword arguments.
        dyn_leaves (list[OpValue]): Flattened dynamic argument leaves.
        dyn_treedef (TreeDef): PyTree structure of `(dyn_pos, dyn_kw)`.
        cfg (_NormalizedJitConfig): Normalized JIT options.
        target_backend (str): Resolved backend identifier.

    Returns:
        _CompiledJitEntry: Compiled graph artifact ready for cached execution.
    """
    increment_trace_count(fun)
    dyn_input_ids: list[str] = []
    traced_leaves: list[OpValue] = []
    with ConfigContext(eager_mode=False, backend=target_backend):
        graph = global_tracing_state.start_tracing(name="JitFunction")
        try:
            for idx, leaf in enumerate(dyn_leaves):
                if isinstance(leaf, Tensor):
                    inp_id = f"jit_arg_{idx}"
                    dyn_input_ids.append(inp_id)
                    str_shape = tuple(int(d) if isinstance(d, int) else str(d) for d in leaf.shape)
                    in_node = LogicalNode(id=inp_id, op_type="Input", inputs=[], shape_metadata=str_shape)
                    global_tracing_state.add_node(in_node)
                    graph.inputs.append(inp_id)
                    proxy = ProxyTensor(id=inp_id, shape=str_shape, dtype=str(leaf.dtype))
                    traced_leaves.append(Tensor(proxy, leaf.config))
                else:
                    traced_leaves.append(leaf)
            unflat = tree_unflatten(dyn_treedef, traced_leaves)
            assert isinstance(unflat, tuple) and len(unflat) == 2
            traced_pos, traced_kw = unflat[0], unflat[1]
            assert isinstance(traced_pos, list) and isinstance(traced_kw, dict)
            call_pos, call_kw = _merge_call_args(total_pos, static_pos, traced_pos, static_kw, traced_kw)
            _callable_fn: OpValue = fun
            assert is_variadic_callable(_callable_fn)
            raw_out = _callable_fn(*call_pos, **call_kw)
            out_leaves_raw, out_treedef = tree_flatten(raw_out)
        finally:
            captured = global_tracing_state.stop_tracing()

    ir_graph = captured if captured is not None else graph
    compiled_leaves: list[_CompiledOutputLeaf] = []
    out_ids: list[str] = []
    for leaf in out_leaves_raw:
        if isinstance(leaf, Tensor) and is_traced_data(leaf.data):
            nid = str(leaf.data.id)
            if nid not in out_ids:
                out_ids.append(nid)
            compiled_leaves.append(_CompiledOutputLeaf(True, nid, leaf.config, None))
        else:
            compiled_leaves.append(_CompiledOutputLeaf(False, "", None, leaf))
    ir_graph.outputs = out_ids
    opt_graph = _optimize_jit_graph(ir_graph, cfg.keep_unused)
    return _CompiledJitEntry(
        graph=opt_graph,
        dyn_input_ids=tuple(dyn_input_ids),
        out_leaves=tuple(compiled_leaves),
        out_treedef=out_treedef,
        backend_name=target_backend,
        device_override=cfg.device,
    )


def _execute_compiled_jit(entry: _CompiledJitEntry, dyn_leaves: list[OpValue]) -> PrimalTree:
    """Evaluate a cached `_CompiledJitEntry` on concrete `dyn_leaves` via the target backend.

    Args:
        entry (_CompiledJitEntry): Cached compiled `LogicalGraph` entry.
        dyn_leaves (list[OpValue]): Flattened dynamic runtime leaves.

    Returns:
        PrimalTree: Reconstructed PyTree of concrete `Tensor` and scalar outputs.
    """
    backend_cls = BackendRegistry.get(entry.backend_name)
    backend_inst = backend_cls(entry.graph)
    tensor_leaves = [leaf for leaf in dyn_leaves if isinstance(leaf, Tensor)]
    active_inputs = set(entry.graph.inputs)
    inputs_map: dict[str, OpValue] = {}
    for inp_id, tensor_val in zip(entry.dyn_input_ids, tensor_leaves):
        if inp_id in active_inputs:
            inputs_map[inp_id] = backend_cls.asarray(tensor_val.data)

    with ConfigContext(eager_mode=True, backend=entry.backend_name):
        eval_outputs = evaluate_graph(entry.graph, inputs_map, backend=backend_inst)

    target_dev = Device(entry.device_override) if entry.device_override is not None else None
    result_leaves: list[OpValue] = []
    for spec in entry.out_leaves:
        if spec.is_node:
            raw_val = eval_outputs[spec.node_id]
            base_cfg = spec.tensor_config if spec.tensor_config is not None else TensorConfig((), DType.Float32, "cpu")
            out_cfg = TensorConfig(
                base_cfg.shape,
                base_cfg.dtype,
                target_dev if target_dev is not None else base_cfg.device,
                requires_grad=base_cfg.requires_grad,
            )
            result_leaves.append(Tensor(raw_val, out_cfg))
        else:
            result_leaves.append(spec.literal_value)
    return as_primal_tree(tree_unflatten(entry.out_treedef, result_leaves))


def _build_jit_cache_key(
    fun: DifferentiableFn,
    static_pos: list[tuple[int, OpValue]],
    static_kw: dict[str, OpValue],
    dyn_leaves: list[OpValue],
    dyn_treedef: TreeDef,
    cfg: _NormalizedJitConfig,
    target_backend: str,
) -> str:
    """Compute a deterministic compilation cache key from static values and dynamic tensor specs.

    Args:
        fun (DifferentiableFn): Wrapped function.
        static_pos (list[tuple[int, OpValue]]): Static positional arguments.
        static_kw (dict[str, OpValue]): Static keyword arguments.
        dyn_leaves (list[OpValue]): Flattened dynamic leaves.
        dyn_treedef (TreeDef): PyTree structure of dynamic arguments.
        cfg (_NormalizedJitConfig): Normalized JIT options.
        target_backend (str): Target backend name.

    Returns:
        str: Canonical string cache key.
    """
    leaf_sigs = tuple(
        ("T", leaf.shape, str(leaf.dtype), str(leaf.device))
        if isinstance(leaf, Tensor)
        else ("S", type(leaf).__name__, repr(leaf))
        for leaf in dyn_leaves
    )
    static_sig = (
        tuple((idx, type(v).__name__, repr(v)) for idx, v in static_pos),
        tuple((k, type(static_kw[k]).__name__, repr(static_kw[k])) for k in sorted(static_kw)),
    )
    return repr((id(fun), static_sig, repr(dyn_treedef), leaf_sigs, target_backend, cfg.device, cfg.keep_unused))


def jit(
    fun: DifferentiableFn,
    options: JitOptions | OptionsJitOptions | None = None,
) -> VariadicCallableProtocol:
    """Wrap `fun` for trace-based IR optimization (`PassManager`), backend lowering, and caching.

    Args:
        fun (DifferentiableFn): Function to JIT compile.
        options (JitOptions | OptionsJitOptions | None): Optional JIT configuration.

    Returns:
        VariadicCallableProtocol: JIT-compiled callable with shape/static-arg specialization cache.
    """
    cfg = _normalize_jit_options(options)

    def wrapped(
        *args: OpValue,
        **kwargs: OpValue,
    ) -> PrimalTree:
        """Execute `fun` via cached `LogicalGraph` compilation or inline when tracing/disabled.

        Args:
            *args (OpValue): Positional call arguments.
            **kwargs (OpValue): Keyword call arguments.

        Returns:
            PrimalTree: Result of evaluating `fun(*args, **kwargs)`.
        """
        _callable_fn_1: OpValue = fun
        assert is_variadic_callable(_callable_fn_1)
        if global_tracing_state.is_tracing:
            return as_primal_tree(_callable_fn_1(*args, **kwargs))
        if is_jit_disabled() or cfg.inline:
            with ConfigContext(eager_mode=True):
                return as_primal_tree(_callable_fn_1(*args, **kwargs))

        static_pos, dyn_pos, static_kw, dyn_kw = _partition_call_args(args, kwargs, cfg)
        dyn_leaves, dyn_treedef = tree_flatten((dyn_pos, dyn_kw))
        if not any(isinstance(leaf, Tensor) for leaf in dyn_leaves):
            return as_primal_tree(_callable_fn_1(*args, **kwargs))

        target_backend = (
            cfg.backend
            if cfg.backend is not None
            else str(config.backend or get_active_backend().backend_name or "numpy")
        )
        cache_key = _build_jit_cache_key(fun, static_pos, static_kw, dyn_leaves, dyn_treedef, cfg, target_backend)
        with _JIT_CACHE_LOCK:
            cached = _JIT_CACHE.get(cache_key)
            if cached is not None:
                _JIT_CACHE.move_to_end(cache_key)
        if cached is None:
            cached = _trace_and_compile_jit(
                fun,
                len(args),
                static_pos,
                static_kw,
                dyn_leaves,
                dyn_treedef,
                cfg,
                target_backend,
            )
            with _JIT_CACHE_LOCK:
                _JIT_CACHE[cache_key] = cached
                while len(_JIT_CACHE) > _MAX_JIT_CACHE_ENTRIES:
                    _JIT_CACHE.popitem(last=False)
        return _execute_compiled_jit(cached, dyn_leaves)

    return wrapped


def _is_symbolic_shape_descriptor(leaf: OpValue) -> bool:
    """Return True if `leaf` is a `Tensor`, `TensorSpec`, or `TensorConfig` for symbolic shape inference.

    Args:
        leaf (OpValue): Candidate PyTree leaf.

    Returns:
        bool: True when `leaf` triggers zero-allocation symbolic `eval_shape`.
    """
    return isinstance(leaf, (Tensor, TensorSpec, TensorConfig))


def _leaf_to_symbolic_tracer(leaf: OpValue, idx: int) -> OpValue:
    """Convert a `Tensor`, `TensorSpec`, or `TensorConfig` leaf into a zero-storage `Tensor(ProxyTensor)`.

    Args:
        leaf (OpValue): Input PyTree leaf.
        idx (int): Leaf index used to form deterministic symbolic input IDs.

    Returns:
        OpValue: Zero-storage symbolic `Tensor` or unchanged non-tensor leaf.
    """
    inp_id = f"eval_shape_in_{idx}"
    if isinstance(leaf, Tensor):
        shape = tuple(int(d) if isinstance(d, int) else str(d) for d in leaf.shape)
        dtype_str = str(leaf.dtype)
        t_cfg = leaf.config
    elif isinstance(leaf, TensorSpec):
        shape = tuple(int(d) if isinstance(d, int) else str(d) for d in leaf.shape)
        dtype_str = str(leaf.dtype)
        t_cfg = TensorConfig(shape, dtype_str, "cpu")
    elif isinstance(leaf, TensorConfig):
        raw_shape = leaf.shape if isinstance(leaf.shape, tuple) else ()
        shape = tuple(int(d) if isinstance(d, int) else str(d) for d in raw_shape)
        dtype_str = str(leaf.dtype) if leaf.dtype is not None else "float32"
        t_cfg = leaf
    else:
        return leaf
    node = LogicalNode(id=inp_id, op_type="Input", inputs=[], shape_metadata=shape)
    global_tracing_state.add_node(node)
    return Tensor(ProxyTensor(id=inp_id, shape=shape, dtype=dtype_str), t_cfg)


def eval_shape(
    fun: DifferentiableFn,
    *args: OpValue,
    **kwargs: OpValue,
) -> PrimalTree:
    """Infer output shapes and dtypes of `fun(*args, **kwargs)` symbolically without buffer allocation.

    When any PyTree leaf in `(args, kwargs)` is a `Tensor`, `TensorSpec`, or `TensorConfig`,
    `fun` is traced symbolically on zero-storage `ProxyTensor` handles and each output tensor
    leaf is returned as an abstract `TensorSpec(shape=..., dtype=...)`. For pure scalar calls
    with no tensor descriptors, `fun(*args, **kwargs)` is evaluated directly.

    Args:
        fun (DifferentiableFn): Function whose output shapes and dtypes are queried.
        *args (OpValue): Positional arguments or abstract shape/dtype descriptors (`TensorSpec`/`TensorConfig`).
        **kwargs (OpValue): Keyword arguments or abstract shape/dtype descriptors.

    Returns:
        PrimalTree: PyTree of `TensorSpec` descriptors (symbolic mode) or direct scalar result.
    """
    _callable_fn_1: OpValue = fun
    assert is_variadic_callable(_callable_fn_1)
    in_leaves, in_treedef = tree_flatten((list(args), dict(kwargs)))
    if not any(_is_symbolic_shape_descriptor(leaf) for leaf in in_leaves):
        return as_primal_tree(_callable_fn_1(*args, **kwargs))

    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing(name="EvalShape")
        try:
            sym_leaves = [_leaf_to_symbolic_tracer(leaf, idx) for idx, leaf in enumerate(in_leaves)]
            unflat = tree_unflatten(in_treedef, sym_leaves)
            assert isinstance(unflat, tuple) and len(unflat) == 2
            sym_args, sym_kwargs = unflat[0], unflat[1]
            assert isinstance(sym_args, list) and isinstance(sym_kwargs, dict)
            raw_out = _callable_fn_1(*sym_args, **sym_kwargs)
            out_leaves, out_treedef = tree_flatten(raw_out)
        finally:
            global_tracing_state.stop_tracing()

    spec_leaves: list[OpValue] = [
        TensorSpec(
            shape=tuple(int(d) if isinstance(d, int) else str(d) for d in leaf.shape),
            dtype=leaf.dtype.value,
        )
        if isinstance(leaf, Tensor)
        else leaf
        for leaf in out_leaves
    ]
    return as_primal_tree(tree_unflatten(out_treedef, spec_leaves))


__all__ = [
    "JitOptions",
    "clear_jit_cache",
    "disable_jit",
    "eval_shape",
    "get_jit_cache_size",
    "is_jit_disabled",
    "jit",
    "set_max_jit_cache_entries",
]
