"""Provides required module functionality."""

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer


def test_base_coverage_brute() -> None:
    """Execute the requested function."""
    config.eager_mode = False

    @register_op("TestCoverageOp")
    class TestCoverageOp(OpDef):
        """Docstring."""

        def infer_shape(self, *args: object, **kwargs: object) -> tuple:
            """Docstring."""
            return ()

        def eager_eval(self, *args: object, **kwargs: object) -> int:
            """Docstring."""
            return 1

    _tracer.start_tracing()
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

    _tracer.stop_tracing()
    config.eager_mode = True
