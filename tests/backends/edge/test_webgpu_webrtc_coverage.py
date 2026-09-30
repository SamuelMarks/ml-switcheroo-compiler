"""Tests for test_webgpu_webrtc_coverage."""

from __future__ import annotations

import multiprocessing
import os
from typing import Callable, Union

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.edge.webgpu_webrtc as webrtc_mod
import ml_switcheroo_compiler.benchmarks.config_models as cfg_models_mod
from ml_switcheroo_compiler.ir.core import IRGraph

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


def test_webgpu_webrtc_emitters() -> None:
    """Verify WebRTC initialization and collective operation JS template generation."""
    # 1. emit_webrtc_init
    init_str = webrtc_mod.emit_webrtc_init()
    assert isinstance(init_str, str)
    assert "RTCPeerConnection" in init_str or len(init_str) > 0

    # 2. emit_webrtc_op for all supported operations
    for op in ["AllReduce", "AllGather", "AllToAll", "ReduceScatter", "Broadcast"]:
        emitted = webrtc_mod.emit_webrtc_op(op, "local_data", "op_1")
        assert len(emitted) > 0

    # Unsupported operation returns empty string
    assert webrtc_mod.emit_webrtc_op("UnknownCollective", "local_data", "op_2") == ""

    # Non-existent yaml paths branches
    with pytest.MonkeyPatch.context() as mp:
        orig_exists = os.path.exists

        def mock_exists_no_top(path: str) -> bool:
            if "webrtc_topology.yaml" in path:
                return False
            return orig_exists(path)

        mp.setattr(os.path, "exists", mock_exists_no_top)
        assert webrtc_mod.emit_webrtc_init() == ""
        assert webrtc_mod.emit_webrtc_op("AllReduce", "local_data", "op_1") == ""

        def mock_exists_no_col(path: str) -> bool:
            if "webrtc_collectives.yaml" in path:
                return False
            return orig_exists(path)

        mp.setattr(os.path, "exists", mock_exists_no_col)
        assert len(webrtc_mod.emit_webrtc_init()) > 0
        assert len(webrtc_mod.emit_webrtc_op("AllReduce", "local_data", "op_1")) > 0
