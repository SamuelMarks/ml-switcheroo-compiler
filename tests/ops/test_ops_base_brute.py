"""Module docstring."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.tracing.tracer import _tracer, ProxyTensor
import numpy as np


def test_base_coverage_brute() -> None:
    """Function docstring."""
    config.eager_mode = False

    @register_op("TestCoverageOp")
    class TestCoverageOp(OpDef):
        def infer_shape(self, *args: object, **kwargs: object) -> tuple:
            return ()

        def numpy_eval(self, *args: object, **kwargs: object) -> int:
            return 1

    _tracer.start_tracing()
    op = TestCoverageOp()

    t1 = Tensor(
        data=ProxyTensor(id="n1", shape=()), shape=(), dtype=DType.Int32, device=Device("cpu")
    )
    t2 = Tensor(
        data=ProxyTensor(id="n2", shape=()), shape=(), dtype=DType.Int32, device=Device("cpu")
    )

    op(t1, t2)

    t3 = Tensor(data=np.array(5), shape=(), dtype=DType.Int32, device=Device("cpu"))
    op(t3)

    op(ProxyTensor(id="n3", shape=()))
    op([1, 2, 3])

    op([1, 2, 3], [4, 5, 6])

    op(t1, dtype=DType.Float32)

    _tracer.stop_tracing()
    config.eager_mode = True
