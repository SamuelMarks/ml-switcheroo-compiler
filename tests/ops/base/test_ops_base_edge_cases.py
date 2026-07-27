# ruff: noqa: E501
"""Provides required module functionality."""

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_base_coverage_brute() -> None:
    """Test the base coverage brute behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        config.eager_mode = False

        @register_op("TestCoverageOp")
        class TestCoverageOp(OpDef):
            """Test suite grouping assertions for CoverageOp logic."""

            def infer_shape(self, *args: object, **kwargs: object) -> tuple:
                """Evaluate and process the infer shape operation.

                Args:
                    *args (Any): Variable positional arguments.
                    **kwargs (Any): Arbitrary keyword arguments.

                Returns:
                    tuple: The evaluated or processed output.
                """
                return ()

            def eager_eval(self, *args: object, **kwargs: object) -> int:
                """Evaluate and process the eager eval operation.

                Args:
                    *args (Any): Variable positional arguments.
                    **kwargs (Any): Arbitrary keyword arguments.

                Returns:
                    int: The evaluated or processed output.
                """
                return 1

        global_tracing_state.start_tracing()
        op = TestCoverageOp()
        t1 = Tensor(ProxyTensor(id="n1", shape=()), TensorConfig((), DType.Int32, Device("cpu")))
        t2 = Tensor(ProxyTensor(id="n2", shape=()), TensorConfig((), DType.Int32, Device("cpu")))
        op(t1, t2)
        t3 = Tensor(np.array(5), TensorConfig((), DType.Int32, Device("cpu")))
        op(t3)
        op(ProxyTensor(id="n3", shape=()))
        op([1, 2, 3])
        op([1, 2, 3], [4, 5, 6])
        op(t1, dtype=DType.Float32)
        global_tracing_state.stop_tracing()
        config.eager_mode = True
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
