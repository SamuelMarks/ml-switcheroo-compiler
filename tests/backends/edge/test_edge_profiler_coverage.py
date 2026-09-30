"""Tests for test_edge_profiler_coverage."""

from __future__ import annotations

import multiprocessing
import sys
from typing import Callable, Union

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.edge.profiler as edge_prof_mod
import ml_switcheroo_compiler.benchmarks.config_models as cfg_models_mod
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


def test_edge_profiler_coverage() -> None:
    """Verify EdgeProfiler timing, memory calculation, and input preparation branches."""
    # 1. _get_process_memory_mb
    mem_mb = edge_prof_mod._get_process_memory_mb()
    assert mem_mb > 0.0

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "platform", "linux")
        assert edge_prof_mod._get_process_memory_mb() > 0.0

    # 2. _calculate_edge_memory_bytes
    empty_graph = IRGraph(name="empty")
    assert edge_prof_mod._calculate_edge_memory_bytes(empty_graph) == 65536

    graph = IRGraph(name="test_edge")
    in_node = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(-1, 10))
    add_node = IRNode(id="out", op_type="Add", inputs=["x", "x"], shape_metadata=(1, 10))
    graph.nodes["x"] = in_node
    graph.nodes["out"] = add_node
    graph.inputs = ["x"]
    graph.outputs = ["out"]

    bytes_alloc = edge_prof_mod._calculate_edge_memory_bytes(graph)
    assert bytes_alloc >= 65536

    # 3. profile_graph
    profiler = edge_prof_mod.EdgeProfiler()
    inputs = {"x": np.ones((1, 10), dtype=np.float32)}
    res = profiler.profile_graph(graph, inputs, num_iters=2, warmup_iters=1)
    assert "latency_ms" in res
    assert res["peak_memory_mb"] > 0.0

    # Evaluation exception fallback branch
    failing_graph = IRGraph(name="failing")
    failing_graph.nodes["bad"] = IRNode(id="bad", op_type="NonexistentOp", inputs=["x"])
    failing_graph.inputs = ["x"]
    res_fallback = profiler.profile_graph(failing_graph, inputs, num_iters=1, warmup_iters=1)
    assert "latency_ms" in res_fallback

    # Mismatched inputs fallback
    mismatched_inputs = {"other_key": np.ones((1, 10), dtype=np.float32)}
    prepared = profiler._prepare_inputs(graph, mismatched_inputs)
    assert len(prepared) == 1
