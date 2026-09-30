"""Tests for test_orchestrator_coverage."""

from __future__ import annotations

import multiprocessing
from typing import Callable, Union

import numpy as np
import pytest

import ml_switcheroo_compiler.benchmarks.config_models as cfg_models_mod
import ml_switcheroo_compiler.benchmarks.orchestrator as orch_mod
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

MetricsVal = Union[list[float], float]

ProcessArg = Union[tuple[str, str], IRGraph, dict[str, np.ndarray], str, int, multiprocessing.Queue, None]


class MockSimpleProfiler:
    """Mock profiler providing profile_graph implementation."""

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, np.ndarray],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, MetricsVal]:
        """Profile graph execution.

        Args:
            graph (IRGraph): Computational graph.
            inputs (dict[str, np.ndarray]): Inputs.
            device (str | None): Device identifier.
            num_iters (int): Measurement count.
            warmup_iters (int): Warmup count.

        Returns:
            dict[str, MetricsVal]: Profiler metrics.
        """
        return {"latencies": [1.0, 2.0], "peak_memory_mb": 10.0}


class MockBackendClass:
    """Mock backend implementation."""

    def compile(self) -> None:
        """Compile backend."""
        pass


class MockDeterministicQueue:
    """Deterministic queue mock avoiding multiprocessing pipe latency."""

    def __init__(self, items: list[tuple[str, Union[dict[str, MetricsVal], str]]]) -> None:
        """Initialize mock queue with payload items.

        Args:
            items (list[tuple[str, Union[dict[str, MetricsVal], str]]]): Items to queue.
        """
        self.items: list[tuple[str, Union[dict[str, MetricsVal], str]]] = list(items)

    def empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            bool: True if empty.
        """
        return len(self.items) == 0

    def get(self) -> tuple[str, Union[dict[str, MetricsVal], str]]:
        """Retrieve next item.

        Returns:
            tuple[str, Union[dict[str, MetricsVal], str]]: Item.
        """
        return self.items.pop(0)

    def put(self, item: tuple[str, Union[dict[str, MetricsVal], str]]) -> None:
        """Put item into queue.

        Args:
            item (tuple[str, Union[dict[str, MetricsVal], str]]): Item.
        """
        self.items.append(item)


class MockProcessOk:
    """Mock multiprocessing process simulating clean execution."""

    def __init__(self, target: Callable[..., None], args: tuple[ProcessArg, ...]) -> None:
        """Initialize mock process.

        Args:
            target (Callable[..., None]): Target function.
            args (tuple[ProcessArg, ...]): Process arguments.
        """
        self._target: Callable[..., None] = target
        self._args: tuple[ProcessArg, ...] = args

    def start(self) -> None:
        """Start process."""
        pass

    def join(self, timeout: int | None = None) -> None:
        """Wait for process completion.

        Args:
            timeout (int | None): Timeout duration in seconds.
        """
        pass

    def is_alive(self) -> bool:
        """Check if process is alive.

        Returns:
            bool: False for completed process.
        """
        return False

    def terminate(self) -> None:
        """Terminate process."""
        pass


class MockProcessTimeout(MockProcessOk):
    """Mock multiprocessing process simulating timeout."""

    def is_alive(self) -> bool:
        """Check if process is alive.

        Returns:
            bool: Always True to simulate timeout.
        """
        return True


def _raise_load_err(p: str = "") -> dict[str, cfg_models_mod.BackendProfile]:
    """Helper to simulate load_backend_profiles error.

    Args:
        p (str): Unused path.

    Returns:
        dict[str, cfg_models_mod.BackendProfile]: Never returned.

    Raises:
        ValueError: Always raised.
    """
    raise ValueError("boom")


def _raise_workload_err() -> dict[str, cfg_models_mod.ModelWorkload]:
    """Helper to simulate load_model_workloads error.

    Returns:
        dict[str, cfg_models_mod.ModelWorkload]: Never returned.

    Raises:
        ValueError: Always raised.
    """
    raise ValueError("cannot load workloads")


class MockDeviceFailingProfiler:
    """Mock profiler that rejects non-None device kwarg and accepts device=None."""

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, np.ndarray],
        *args: int,
        **kwargs: str | int | None,
    ) -> dict[str, MetricsVal]:
        """Profile graph execution with TypeError when device is provided.

        Args:
            graph (IRGraph): Computational graph.
            inputs (dict[str, np.ndarray]): Inputs.
            *args (int): Additional positional args.
            **kwargs (str | int | None): Keyword arguments.

        Returns:
            dict[str, MetricsVal]: Profiler metrics.

        Raises:
            TypeError: When device is not None.
        """
        if kwargs.get("device") is not None:
            raise TypeError("Device argument not accepted")
        return {"latencies": [0.0], "peak_memory_mb": 0.0}


def test_build_ir_graph_from_workload() -> None:
    """Verify construction of LogicalGraph from declarative workload specifications."""
    workload = cfg_models_mod.ModelWorkload(
        description="test model",
        batch_sizes=[1, 4],
        inputs={"in_x": cfg_models_mod.WorkloadTensorSpec(shape=[1, 10], dtype="float32")},
        weights={"w": cfg_models_mod.WorkloadTensorSpec(shape=[10, 5], dtype="float32")},
        nodes=[
            cfg_models_mod.WorkloadNodeSpec(
                id="matmul_0",
                op_type="MatMul",
                inputs=["in_x", "w"],
                attributes={"transpose_a": False},
            )
        ],
        outputs=["matmul_0"],
    )
    graph = cfg_models_mod.build_ir_graph_from_workload(workload, name="test_graph")
    assert graph.name == "test_graph"
    assert "in_x" in graph.nodes
    assert "w" in graph.nodes
    assert "matmul_0" in graph.nodes
    assert graph.outputs == ["matmul_0"]


def test_orchestrator_isolated_worker_and_helpers() -> None:
    """Verify _isolated_worker execution and error queue paths."""
    q: multiprocessing.Queue = multiprocessing.Queue()
    graph = IRGraph(name="iso_graph")

    # 1. Successful worker run
    orch_mod._isolated_worker(
        ("ml_switcheroo_compiler.backends.edge.profiler", "EdgeProfiler"),
        graph,
        {},
        None,
        1,
        1,
        q,
    )
    status, payload = q.get()
    assert status == "ok"
    assert isinstance(payload, dict)

    # 2. Failing worker run
    orch_mod._isolated_worker(
        ("nonexistent.module", "FakeClass"),
        graph,
        {},
        None,
        1,
        1,
        q,
    )
    err_status, err_payload = q.get()
    assert err_status == "error"
    assert "No module named" in str(err_payload)


def test_orchestrator_profiler_and_isolated_branches() -> None:
    """Verify profiler retrieval, isolation execution, and error handling in orchestrator."""
    plan = cfg_models_mod.BenchmarkPlan(
        name="test_plan",
        models=["model_a"],
        batch_sizes=[1],
        targets=[cfg_models_mod.BenchmarkTarget(backend="numpy")],
        num_iterations=2,
        warmup_iterations=1,
    )
    orchestrator = orch_mod.BenchmarkOrchestrator(plan)

    # 1. _load_manifest_profiles including branch with profile missing module
    manifest_specs = orch_mod.BenchmarkOrchestrator._load_manifest_profiles()
    assert len(manifest_specs) > 0

    with pytest.MonkeyPatch.context() as mp:
        no_module_profile = cfg_models_mod.BackendProfile(
            timer="perf_counter",
            sync_method="none",
            default_warmup_iterations=1,
            default_measured_iterations=1,
            memory_tracking="none",
            profiler_module=None,
            profiler_class=None,
        )
        mp.setattr(orch_mod, "load_backend_profiles", lambda p: {"no_mod_be": no_module_profile})
        specs = orch_mod.BenchmarkOrchestrator._load_manifest_profiles()
        assert "no_mod_be" not in specs

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orch_mod.pathlib.Path, "exists", lambda self: False)
        assert orch_mod.BenchmarkOrchestrator._load_manifest_profiles() == {}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orch_mod, "load_backend_profiles", _raise_load_err)
        assert orch_mod.BenchmarkOrchestrator._load_manifest_profiles() == {}

    # 2. _get_profiler for numpy and unsupported and import error
    p_numpy = orchestrator._get_profiler("numpy")
    assert hasattr(p_numpy, "profile_graph")

    with pytest.raises(BackendNotSupportedError, match="does not have a supported profiler"):
        orchestrator._get_profiler("unsupported_backend_xyz")

    with pytest.MonkeyPatch.context() as mp:
        mock_prof_model = cfg_models_mod.BackendProfile(
            timer="perf_counter",
            sync_method="none",
            default_warmup_iterations=1,
            default_measured_iterations=1,
            memory_tracking="none",
            profiler_module="nonexistent.import_err_module",
            profiler_class="SomeClass",
        )
        mp.setattr(orch_mod, "load_backend_profiles", lambda: {"fail_be": mock_prof_model})
        with pytest.raises(BackendNotSupportedError, match="could not be imported"):
            orchestrator._get_profiler("fail_be")

    # 3. _run_single with unregistered backend
    graph = IRGraph(name="test")
    with pytest.raises(BackendNotSupportedError, match="is not registered"):
        orchestrator._run_single(graph, "unknown_backend", batch_size=1, num_iters=1, warmup_iters=1)

    # 4. _run_single with registered backend and weight attributes
    weight_node = IRNode(id="w", op_type="Input", inputs=[], attributes={"shape": [10, 5], "is_weight": True})
    graph.nodes["w"] = weight_node
    graph.inputs = ["w"]
    BackendRegistry.register("mock_be", MockBackendClass)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator, "_get_profiler", lambda be: MockSimpleProfiler())
        res = orchestrator._run_single(graph, "mock_be", batch_size=2, num_iters=1, warmup_iters=1, isolate=False)
        assert res["peak_memory_mb"] == 10.0

        # graph with empty inputs fallback to "in_0"
        empty_graph = IRGraph(name="empty_inputs")
        res_empty = orchestrator._run_single(empty_graph, "mock_be", batch_size=2, num_iters=1, warmup_iters=1)
        assert res_empty["peak_memory_mb"] == 10.0

    # 5. _run_isolated success and error branches
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orch_mod.multiprocessing, "get_context", lambda: orch_mod.multiprocessing)
        mp.setattr(orch_mod.multiprocessing, "Process", MockProcessOk)
        mp.setattr(
            orch_mod.multiprocessing,
            "Queue",
            lambda: MockDeterministicQueue([("ok", {"latencies": [1.0], "peak_memory_mb": 5.0})]),
        )
        metrics = orchestrator._run_isolated(graph, "numpy", {})
        assert metrics["latencies"] == [1.0]

        # _run_single with isolate=True
        single_isolated = orchestrator._run_single(graph, "mock_be", batch_size=1, num_iters=1, warmup_iters=1, isolate=True)
        assert single_isolated["latencies"] == [1.0]

        # Mock isolated run with error payload
        mp.setattr(
            orch_mod.multiprocessing,
            "Queue",
            lambda: MockDeterministicQueue([("error", "subprocess failed")]),
        )
        with pytest.raises(RuntimeError, match="Isolated execution failed"):
            orchestrator._run_isolated(graph, "numpy", {})

        # Mock isolated run with empty queue
        mp.setattr(orch_mod.multiprocessing, "Queue", lambda: MockDeterministicQueue([]))
        with pytest.raises(RuntimeError, match="terminated unexpectedly"):
            orchestrator._run_isolated(graph, "numpy", {})

        # Mock isolated run with timeout
        mp.setattr(orch_mod.multiprocessing, "Process", MockProcessTimeout)
        with pytest.raises(RuntimeError, match="timed out"):
            orchestrator._run_isolated(graph, "numpy", {})


def test_orchestrator_execute_full() -> None:
    """Verify orchestrator full execute loop with fallback IRGraph, TypeError fallback, and results."""
    plan = cfg_models_mod.BenchmarkPlan(
        name="full_plan",
        models=["unregistered_model", "mlp"],
        batch_sizes=[2],
        targets=[cfg_models_mod.BenchmarkTarget(backend="numpy", device="cpu")],
        num_iterations=2,
        warmup_iterations=1,
    )
    orchestrator = orch_mod.BenchmarkOrchestrator(plan)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator, "_get_profiler", lambda be: MockSimpleProfiler())
        results = orchestrator.execute()
        assert len(results) == 2
        res = results[0]
        assert res.model == "unregistered_model"
        assert res.batch_size == 2
        assert res.backend == "numpy"
        assert res.throughput_items_per_sec > 0.0

    # Test TypeError fallback when profiler does not accept device kwarg
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator, "_get_profiler", lambda be: MockDeviceFailingProfiler())
        results_nodevice = orchestrator.execute()
        assert len(results_nodevice) == 2
        assert results_nodevice[0].throughput_items_per_sec == 0.0

    # Test load_model_workloads exception branch in execute
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator, "_get_profiler", lambda be: MockSimpleProfiler())
        mp.setattr(cfg_models_mod, "load_model_workloads", _raise_workload_err)
        results_fallback = orchestrator.execute()
        assert len(results_fallback) == 2
