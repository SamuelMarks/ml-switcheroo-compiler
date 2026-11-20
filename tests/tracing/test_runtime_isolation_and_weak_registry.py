"""Unit tests for ContextVar thread/async isolation, O(1) weak Tensor registry, and bounded callbacks."""

from __future__ import annotations

import asyncio
import gc
import threading
from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.tracing.tracer as tracer_mod
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import backward
from ml_switcheroo_compiler.ir.callback_registry import (
    CustomRuleHandle,
    callback_registry_scope,
    clear_custom_rule_handles,
    get_registered_callback_count,
    register_custom_rule_handle,
    resolve_custom_rule_handle,
    unregister_custom_rule_handle,
)
from ml_switcheroo_compiler.tracing.autograph import LoopOptions
from ml_switcheroo_compiler.tracing.state import (
    TracingState,
    get_live_tensors,
    global_tracing_state,
    live_tensor_scope,
)
from ml_switcheroo_compiler.tracing.tracer import (
    ProxyTensor,
    TracerTape,
    get_trace_count,
    increment_trace_count,
    reset_trace_count,
)


def test_config_and_tracing_spawned_thread_isolation() -> None:
    """Verify spawned OS threads initialize clean ConfigState and TracingState without LookupError."""
    errors: list[BaseException] = []
    results: dict[str, tuple[str, bool, str]] = {}

    def _worker(tag: str, backend_choice: str) -> None:
        """Execute isolated config mutations and graph tracing inside a worker thread.

        Args:
            tag (str): Worker identifier.
            backend_choice (str): Backend string to set locally.
        """
        try:
            config.backend = backend_choice
            config.seed = 99
            config.default_float_dtype = DType.Float64
            config.default_int_dtype = DType.Int32
            config.default_device = Device("cpu")
            config.current_stream = f"stream_{tag}"
            config.layout_map = {"x": "NHWC"}
            config.enable_x64 = False
            config.op_determinism = True
            config.tensor_float_32_execution = True
            with ConfigContext(eager_mode=False):
                g = global_tracing_state.start_tracing(name=f"Graph_{tag}")
                global_tracing_state.add_node(LogicalNode(id=f"n_{tag}", op_type="Input"))
                finished = global_tracing_state.stop_tracing()
                assert finished is g
            results[tag] = (config.backend, global_tracing_state.is_tracing, g.name)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    orig_backend = config.backend
    t1 = threading.Thread(target=_worker, args=("w1", "pure_python"))
    t2 = threading.Thread(target=_worker, args=("w2", "numpy"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
    assert results["w1"] == ("pure_python", False, "Graph_w1")
    assert results["w2"] == ("numpy", False, "Graph_w2")
    assert config.backend == orig_backend
    assert not global_tracing_state.is_tracing


def test_asyncio_task_copy_on_write_isolation() -> None:
    """Verify concurrent asyncio tasks do not corrupt sibling Config or TracingState frames."""

    async def _run_concurrent() -> tuple[tuple[str, list[str]], tuple[str, list[str]]]:
        """Spawn two interleaved async tasks that trace distinct graphs concurrently.

        Returns:
            tuple[tuple[str, list[str]], tuple[str, list[str]]]: Captured `(name, node_ids)` per task.
        """
        barrier = asyncio.Event()

        async def _coro_a() -> tuple[str, list[str]]:
            """Trace task A before and after yielding across the event loop.

            Returns:
                tuple[str, list[str]]: Graph name and recorded node IDs.
            """
            config.eager_mode = False
            g_a = global_tracing_state.start_tracing("AsyncA")
            global_tracing_state.add_node(LogicalNode(id="a_0", op_type="Input"))
            await asyncio.sleep(0)
            barrier.set()
            await asyncio.sleep(0)
            global_tracing_state.add_node(LogicalNode(id="a_1", op_type="Relu", inputs=["a_0"]))
            out_a = global_tracing_state.stop_tracing()
            assert out_a is g_a
            return (g_a.name, list(g_a.nodes.keys()))

        async def _coro_b() -> tuple[str, list[str]]:
            """Trace task B concurrently while task A is still suspended mid-trace.

            Returns:
                tuple[str, list[str]]: Graph name and recorded node IDs.
            """
            await barrier.wait()
            config.eager_mode = False
            g_b = global_tracing_state.start_tracing("AsyncB")
            global_tracing_state.add_node(LogicalNode(id="b_0", op_type="Input"))
            out_b = global_tracing_state.stop_tracing()
            assert out_b is g_b
            return (g_b.name, list(g_b.nodes.keys()))

        res_a, res_b = await asyncio.gather(_coro_a(), _coro_b())
        return res_a, res_b

    res_a, res_b = asyncio.run(_run_concurrent())
    assert res_a == ("AsyncA", ["a_0", "a_1"])
    assert res_b == ("AsyncB", ["b_0"])


def test_tracing_state_setters_deleters_and_weak_trace_counts() -> None:
    """Cover TracingState property setters/deleters, TracerTape facade, and bounded trace counters."""
    g = LogicalGraph_from_state(TracingState())
    lo = LoopOptions(parallel_iterations=2)
    for attr_name, patched_val in (
        ("is_tracing", True),
        ("active_graph", g),
        ("graph_stack", [g]),
        ("constant_cache", {1: "c1"}),
        ("current_loop_options", lo),
        ("loop_options_stack", [lo]),
    ):
        inst = TracingState()
        with patch.object(inst, attr_name, patched_val):
            assert (
                inst.is_tracing
                or inst.active_graph is not None
                or inst.graph_stack
                or inst.constant_cache
                or inst.current_loop_options is not None
                or inst.loop_options_stack
            )
        assert not inst.is_tracing
        assert inst.active_graph is None
        assert inst.graph_stack == []
        assert inst.constant_cache == {}
        assert inst.current_loop_options is None
        assert inst.loop_options_stack == []

    tape = TracerTape()
    tg = tape.start_tracing("TapeGraph")
    tape.add_node(LogicalNode(id="tn1", op_type="Input"))
    assert tape.stop_tracing() is tg

    def _ephemeral() -> int:
        """Return constant scalar for weak trace count test.

        Returns:
            int: Scalar `1`.
        """
        return 1

    increment_trace_count(_ephemeral)
    assert get_trace_count(_ephemeral) == 1
    reset_trace_count(_ephemeral)
    assert get_trace_count(_ephemeral) == 0

    saved_limit = tracer_mod._MAX_FALLBACK_TRACE_ENTRIES
    try:
        tracer_mod._MAX_FALLBACK_TRACE_ENTRIES = 2
        increment_trace_count(101)
        increment_trace_count(202)
        increment_trace_count(303)
        assert get_trace_count(101) == 0
        assert get_trace_count(303) == 1
        reset_trace_count(303)
        assert get_trace_count(303) == 0
    finally:
        tracer_mod._MAX_FALLBACK_TRACE_ENTRIES = saved_limit


def LogicalGraph_from_state(st: TracingState) -> LogicalGraph:
    """Create and stop a single trace on `st` returning the captured `LogicalGraph`.

    Args:
        st (TracingState): State instance to exercise.

    Returns:
        LogicalGraph: Captured graph instance.
    """
    g: LogicalGraph = st.start_tracing("Helper")
    st.stop_tracing()
    return g


def test_weak_tensor_registry_gc_and_zero_gc_get_objects_in_backward() -> None:
    """Verify dead Tensors leave WeakSet immediately and backward() never calls gc.get_objects."""
    outer_tensor = Tensor(np.array(1.0, dtype=np.float32), TensorConfig((), DType.Float32, "cpu"))
    assert any(t is outer_tensor for t in get_live_tensors())

    with live_tensor_scope() as scoped:
        ephemeral = Tensor(np.array(3.0, dtype=np.float32), TensorConfig((), DType.Float32, "cpu"))
        assert id(ephemeral) in scoped
        assert len(get_live_tensors()) == 1
        del ephemeral
        gc.collect()
        assert get_live_tensors() == []

    with ConfigContext(eager_mode=False, backend="numpy"):
        g = global_tracing_state.start_tracing("NoGcBackward")
        try:
            in_node = LogicalNode(id="x_in", op_type="Input", inputs=[], shape_metadata=())
            global_tracing_state.add_node(in_node)
            g.inputs.append("x_in")
            px = ProxyTensor(id="x_in", shape=(), dtype="float32")
            px.concrete_value = np.array(4.0, dtype=np.float32)
            x = Tensor(px, TensorConfig((), DType.Float32, "cpu", requires_grad=True))
            y = x * x
            with live_tensor_scope([x, y]):
                with patch("gc.get_objects", side_effect=AssertionError("gc.get_objects is banned")):
                    backward(y)
            assert x.grad is not None
            assert float(np.asarray(x.grad)) == pytest.approx(8.0)
        finally:
            global_tracing_state.stop_tracing()


def test_callback_registry_scope_eviction_and_unregister() -> None:
    """Verify scoped isolation, LRU eviction, unregistration, and validation errors on callbacks."""
    with pytest.raises(ValueError, match="max_entries must be >= 1"):
        with callback_registry_scope(max_entries=0):
            _ = 0  # pragma: no cover

    with pytest.raises(ValueError, match="max_entries must be >= 1"):
        register_custom_rule_handle(lambda x: x, max_entries=0)

    from ml_switcheroo_compiler.grad.utils import DifferentiableFn

    non_callable_val: DifferentiableFn | None = None
    assert register_custom_rule_handle(non_callable_val) is None

    with callback_registry_scope():
        assert get_registered_callback_count() == 0
        h1 = register_custom_rule_handle(lambda x: x + 1, kind="vjp", max_entries=2)
        h2 = register_custom_rule_handle(lambda x: x + 2, kind="vjp", max_entries=2)
        assert h1 is not None and h2 is not None
        assert resolve_custom_rule_handle(h1)(10) == 11
        h3 = register_custom_rule_handle(lambda x: x + 3, kind="vjp", max_entries=2)
        assert h3 is not None
        assert get_registered_callback_count() == 2
        with pytest.raises(KeyError, match="cannot cross process boundaries"):
            resolve_custom_rule_handle(h2)
        assert unregister_custom_rule_handle(h1) is True
        assert unregister_custom_rule_handle(h1) is False
        assert get_registered_callback_count() == 1
        clear_custom_rule_handles()
        assert get_registered_callback_count() == 0

    with pytest.raises(KeyError, match="not registered"):
        resolve_custom_rule_handle(h3)

    import ml_switcheroo_compiler.ir.callback_registry as cb_mod

    saved_global = dict(cb_mod._CALLBACK_TABLE)
    try:
        clear_custom_rule_handles()
        assert get_registered_callback_count() == 0
        g1 = register_custom_rule_handle(lambda x: x, max_entries=1)
        g2 = register_custom_rule_handle(lambda x: x, max_entries=1)
        assert g1 is not None and g2 is not None
        assert get_registered_callback_count() == 1
        assert unregister_custom_rule_handle(g2.rule_id) is True
        assert unregister_custom_rule_handle("missing_id") is False
        assert isinstance(CustomRuleHandle(rule_id="r"), CustomRuleHandle)
    finally:
        cb_mod._CALLBACK_TABLE.update(saved_global)
