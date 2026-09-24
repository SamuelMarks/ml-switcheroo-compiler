"""Exhaustive tests targeting 100% coverage uplift for profilers, collectives, and eager backends."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import awkward as ak
import pyarrow.compute as pc
import pytest

import ml_switcheroo_compiler.backends.awkward.eager as ak_eager
import ml_switcheroo_compiler.backends.cuda.nccl_collectives as nccl_mod
import ml_switcheroo_compiler.backends.dask.profiler as dask_prof_mod
import ml_switcheroo_compiler.backends.keras.profiler as keras_prof_mod
import ml_switcheroo_compiler.backends.pyarrow_compute.eager as pa_eager
import ml_switcheroo_compiler.backends.rocm.rccl_collectives as rccl_mod
import ml_switcheroo_compiler.backends.tensorflow.profiler as tf_prof_mod
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph


def test_dask_profiler_coverage_uplift() -> None:
    """Exercise missing branches in Dask profiler: da is None, failed compute, and import errors."""
    # 1. _sync_dask_result when da is None
    orig_da = dask_prof_mod.da
    try:
        dask_prof_mod.da = None
        assert dask_prof_mod._sync_dask_result("dummy_value") == "dummy_value"
    finally:
        dask_prof_mod.da = orig_da

    # 2. _sync_dask_result when compute() raises an exception
    class FailingCompute:
        """Mock container whose compute method raises an exception."""

        def compute(self) -> None:
            """Raise exception to test error handling."""
            raise RuntimeError("Compute failure test")

    fail_obj = FailingCompute()
    assert dask_prof_mod._sync_dask_result(fail_obj) is fail_obj

    # 3. Dask profiler import error branch
    with patch.dict(sys.modules, {"dask.array": None}):
        importlib.reload(dask_prof_mod)
        assert dask_prof_mod.da is None

    importlib.reload(dask_prof_mod)


def test_keras_profiler_coverage_uplift() -> None:
    """Exercise missing branches in Keras profiler: keras is None, failed numpy(), empty inputs, and import errors."""
    # 1. _sync_keras_result when keras is None
    orig_keras = keras_prof_mod.keras
    try:
        keras_prof_mod.keras = None
        assert keras_prof_mod._sync_keras_result("dummy") is None
        prof = keras_prof_mod.KerasProfiler()
        assert prof._prepare_inputs(IRGraph(), {"x": [1.0]}) == []
    finally:
        keras_prof_mod.keras = orig_keras

    # 2. _sync_keras_result when .numpy() raises an exception
    class FailingNumpy:
        """Mock tensor whose numpy method raises an exception."""

        def numpy(self) -> None:
            """Raise exception to test error handling."""
            raise RuntimeError("Numpy conversion error")

    keras_prof_mod._sync_keras_result(FailingNumpy())

    # 3. _build_step_fn when keras_inputs is empty
    prof_inst = keras_prof_mod.KerasProfiler()
    step_fn_empty = prof_inst._build_step_fn(has_ops=False, compiled_model=None, keras_inputs=[])
    res_empty = step_fn_empty()
    assert res_empty is not None

    # 4. Keras profiler import error branch
    with patch.dict(sys.modules, {"keras": None}):
        importlib.reload(keras_prof_mod)
        assert keras_prof_mod.keras is None

    importlib.reload(keras_prof_mod)


def test_tensorflow_profiler_coverage_uplift() -> None:
    """Exercise import error branch for TensorFlow profiler."""
    with patch.dict(sys.modules, {"tensorflow": None}):
        importlib.reload(tf_prof_mod)
        assert tf_prof_mod.tf is None

    importlib.reload(tf_prof_mod)


def test_nccl_and_rccl_collectives_free_errors() -> None:
    """Exercise exception handling and allocated buffer branches in free_buffer for NCCL and RCCL."""
    # 1. NCCL cudaFree error
    nccl_inst = nccl_mod.NCCLDriver()
    mock_cuda_rt = MagicMock()
    mock_cuda_rt.cudaFree.side_effect = RuntimeError("cudaFree failed")
    nccl_inst.cuda_rt = mock_cuda_rt

    # Trigger try/except pass
    nccl_inst.free_buffer(123456)

    # Trigger else: pop from _allocated_buffers
    nccl_inst._allocated_buffers[999] = "dummy_buf"
    nccl_inst.free_buffer(999)
    assert 999 not in nccl_inst._allocated_buffers

    # 2. RCCL hipFree error
    rccl_inst = rccl_mod.RCCLDriver()
    mock_hip_rt = MagicMock()
    mock_hip_rt.hipFree.side_effect = RuntimeError("hipFree failed")
    rccl_inst.hip_rt = mock_hip_rt

    # Trigger try/except pass
    rccl_inst.free_buffer(654321)

    # Trigger else: pop from _allocated_buffers
    rccl_inst._allocated_buffers[888] = "dummy_buf"
    rccl_inst.free_buffer(888)
    assert 888 not in rccl_inst._allocated_buffers


def test_pyarrow_compute_eager_error_branch() -> None:
    """Exercise error branch when target_name in ARROW_COMPUTE_OP_MAP raises an exception."""
    with patch.object(pc, "negate", side_effect=RuntimeError("Arrow compute failure")):
        with pytest.raises(BackendNotSupportedError, match="Failed executing pyarrow.compute"):
            pa_eager.execute_op("Neg", 1)


def test_awkward_eager_coverage_uplift() -> None:
    """Exercise Tensor unwrapping, import error, and execution failure in Awkward eager."""

    # 1. _unwrap_arg with object named Tensor and data attribute
    class Tensor:
        """Mock Tensor container."""

        def __init__(self, data: object) -> None:
            """Initialize with data payload.

            Args:
                data (object): Data payload.
            """
            self.data = data

    t = Tensor(data=[10, 20, 30])
    assert ak_eager._unwrap_arg(t) == [10, 20, 30]

    # 2. Awkward op execution failure raising BackendNotSupportedError
    with patch.object(ak, "sum", side_effect=RuntimeError("Awkward execution error")):
        with pytest.raises(BackendNotSupportedError, match="Failed executing awkward"):
            ak_eager.execute_op("Sum", [1, 2, 3])

    # 3. Awkward import failure branch (lines 143-144)
    with patch.dict(sys.modules, {"awkward": None}):
        res = ak_eager.execute_op("Add", 1, 2)
        assert res == 3
