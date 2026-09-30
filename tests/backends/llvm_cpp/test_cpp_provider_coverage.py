"""Tests for test_cpp_provider_coverage."""

from __future__ import annotations

import multiprocessing
from typing import Callable, Union

import numpy as np

import ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider as cpp_prov_mod
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


def test_cpp_provider_complete() -> None:
    """Verify C++ provider templates, operations, and mathematical syntheses."""
    # 1. Prelude
    prelude = cpp_prov_mod.get_cpp_prelude()
    assert isinstance(prelude, str)
    assert len(prelude) > 0

    # 2. Template retrieval
    tmpl_valid = cpp_prov_mod.get_cpp_template("unary")
    assert "template" in tmpl_valid or len(tmpl_valid) > 0

    tmpl_invalid = cpp_prov_mod.get_cpp_template("nonexistent_template_xyz")
    assert tmpl_invalid == {}

    # 3. get_cpp_operation registered vs synthesized
    op_sin = cpp_prov_mod.get_cpp_operation("sin")
    assert op_sin is not None

    # Operation in CPP_SYNTH_MATH_MAP but not in declared yaml operations
    op_cube = cpp_prov_mod.get_cpp_operation("cube")
    assert op_cube is not None
    assert "in0_val * in0_val * in0_val" in op_cube.scalar_expr

    # synthesize with allow_synth
    op_synth_unary = cpp_prov_mod.get_cpp_operation("custom_unary", num_inputs=1, allow_synth=True)
    assert op_synth_unary is not None
    assert op_synth_unary.template == "unary"

    op_synth_binary = cpp_prov_mod.get_cpp_operation("custom_binary", num_inputs=2, allow_synth=True)
    assert op_synth_binary is not None
    assert op_synth_binary.template == "binary"

    op_synth_ternary = cpp_prov_mod.get_cpp_operation("custom_ternary", num_inputs=3, allow_synth=True)
    assert op_synth_ternary is not None
    assert op_synth_ternary.template == "ternary"

    # Explicit scalar expression synthesis
    op_custom_expr = cpp_prov_mod.synthesize_cpp_operation("foo", scalar_expr="in0_val * in2_val")
    assert op_custom_expr.template == "ternary"

    op_custom_binary_expr = cpp_prov_mod.synthesize_cpp_operation("bar", scalar_expr="in0_val * in1_val")
    assert op_custom_binary_expr.template == "binary"

    op_default_ternary = cpp_prov_mod.synthesize_cpp_operation("ternary_default", num_inputs=3)
    assert op_default_ternary.template == "ternary"

    # Unmapped without allow_synth
    assert cpp_prov_mod.get_cpp_operation("completely_unknown_op", allow_synth=False) is None
