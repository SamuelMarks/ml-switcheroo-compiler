"""Tests for host-level unified collective dispatch and abstract IR operations."""

from unittest.mock import MagicMock, patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.distributed.host_collectives import (
    HostCollectiveCommunicator,
    _dispatch_eager_collective,
    _dispatch_hardware_collective,
    _dispatch_jax_collective,
    _dispatch_numpy_collective,
    _dispatch_pytorch_collective,
    all_gather,
    all_reduce,
    all_to_all,
    broadcast,
    reduce_scatter,
)


def test_host_collectives_symbolic_ir_emission() -> None:
    """Verify collective functions emit abstract IR nodes in symbolic tracing mode."""
    old_eager = config.eager_mode
    try:
        config.eager_mode = False

        t = Tensor(np.zeros((4, 4), dtype=np.float32), TensorConfig(shape=(4, 4), dtype="float32", device="cpu"))

        ar = all_reduce(t, op="SUM", mesh_axis="data")
        assert isinstance(ar, Tensor)
        assert ar.shape == (4, 4)

        ag = all_gather(t, axis=0, mesh_axis="data", world_size=2)
        assert isinstance(ag, Tensor)
        assert ag.shape == (8, 4)

        rs = reduce_scatter(t, op="SUM", scatter_dim=0, mesh_axis="data", world_size=2)
        assert isinstance(rs, Tensor)
        assert rs.shape == (2, 4)

        a2a = all_to_all(t, scatter_dim=0, gather_dim=1, mesh_axis="data")
        assert isinstance(a2a, Tensor)
        assert a2a.shape == (4, 4)

        bc = broadcast(t, root=0, mesh_axis="data")
        assert isinstance(bc, Tensor)
        assert bc.shape == (4, 4)

    finally:
        config.eager_mode = old_eager


def test_host_collectives_symbolic_invalid_axis_and_scatter_dim() -> None:
    """Verify symbolic collective handling when axis or scatter_dim is outside shape bounds."""
    old_eager = config.eager_mode
    try:
        config.eager_mode = False
        t = Tensor(np.zeros((4, 4), dtype=np.float32), TensorConfig(shape=(4, 4), dtype="float32", device="cpu"))

        ag = all_gather(t, axis=10, mesh_axis="data", world_size=2)
        assert isinstance(ag, Tensor)
        assert ag.shape == (4, 4)

        rs = reduce_scatter(t, op="SUM", scatter_dim=10, mesh_axis="data", world_size=2)
        assert isinstance(rs, Tensor)
        assert rs.shape == (4, 4)
    finally:
        config.eager_mode = old_eager


def test_host_collectives_eager_numpy_dispatch() -> None:
    """Verify collective functions evaluate numerically in eager mode using host emulator."""
    old_eager = config.eager_mode
    try:
        config.eager_mode = True

        arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ranks_data = [arr1, arr2]

        t1 = Tensor(arr1, TensorConfig(shape=(3,), dtype="float32", device="cpu"))

        # AllReduce
        ar_res = all_reduce(t1, op="SUM", all_ranks_data=ranks_data)
        assert isinstance(ar_res, Tensor)
        np.testing.assert_allclose(ar_res.data, np.array([5.0, 7.0, 9.0]))

        # AllGather
        ag_res = all_gather(t1, axis=0, all_ranks_data=ranks_data)
        assert isinstance(ag_res, Tensor)
        assert ag_res.shape == (6,)

        # ReduceScatter
        rs_res = reduce_scatter(t1, op="SUM", scatter_dim=0, all_ranks_data=ranks_data)
        assert isinstance(rs_res, Tensor)

        # AllToAll
        arr_2d_1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr_2d_2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        t_2d = Tensor(arr_2d_1, TensorConfig(shape=(2, 2), dtype="float32", device="cpu"))
        a2a_res = all_to_all(t_2d, scatter_dim=0, gather_dim=0, all_ranks_data=[arr_2d_1, arr_2d_2])
        assert isinstance(a2a_res, Tensor)

        # Broadcast
        bc_res = broadcast(t1, root=1, all_ranks_data=ranks_data)
        assert isinstance(bc_res, Tensor)
        np.testing.assert_allclose(bc_res.data, arr2)

        # Raw numpy array inputs
        raw_ar = all_reduce(arr1, op="SUM", all_ranks_data=ranks_data)
        assert isinstance(raw_ar, np.ndarray)
        np.testing.assert_allclose(raw_ar, np.array([5.0, 7.0, 9.0]))

        raw_ag = all_gather(arr1, axis=0, all_ranks_data=ranks_data)
        assert isinstance(raw_ag, np.ndarray)

        raw_rs = reduce_scatter(arr1, op="SUM", scatter_dim=0, all_ranks_data=ranks_data)
        assert isinstance(raw_rs, np.ndarray)

        raw_a2a = all_to_all(arr_2d_1, scatter_dim=0, gather_dim=0, all_ranks_data=[arr_2d_1, arr_2d_2])
        assert isinstance(raw_a2a, np.ndarray)

        raw_bc = broadcast(arr1, root=0, all_ranks_data=ranks_data)
        assert isinstance(raw_bc, np.ndarray)

    finally:
        config.eager_mode = old_eager


def test_host_collectives_backend_routing() -> None:
    """Verify dynamic routing via BackendRegistry for cuda, rocm, pytorch, jax."""
    old_eager = config.eager_mode
    try:
        config.eager_mode = True
        inp = np.array([1.0, 2.0], dtype=np.float32)

        # CUDA backend mock
        cuda_backend = MagicMock()
        cuda_backend.name = "cuda"
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=cuda_backend):
            with patch("ml_switcheroo_compiler.backends.cuda.nccl_collectives.NCCLDriver.is_available", return_value=True):
                with patch("ml_switcheroo_compiler.distributed.host_collectives._dispatch_accelerator_collective", return_value=inp):
                    from ml_switcheroo_compiler.distributed.host_collectives import _dispatch_eager_collective

                    res = _dispatch_eager_collective("AllReduce", inp, op="SUM")
                    np.testing.assert_allclose(res, inp)

        # ROCm backend mock
        rocm_backend = MagicMock()
        rocm_backend.name = "rocm"
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=rocm_backend):
            with patch("ml_switcheroo_compiler.backends.rocm.rccl_collectives.RCCLDriver.is_available", return_value=True):
                res = all_reduce(inp, op="SUM")
                np.testing.assert_allclose(res, inp)

        # PyTorch backend mock
        pt_backend = MagicMock()
        pt_backend.name = "pytorch"
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=pt_backend):
            res = all_reduce(inp, op="SUM")
            assert res is not None
            assert all_gather(inp, axis=0) is not None
            assert reduce_scatter(inp, op="SUM", scatter_dim=0) is not None
            assert broadcast(inp, root=0) is not None

        # PyTorch backend fallback when pytorch collective returns None (line 552)
        pt_backend = MagicMock()
        pt_backend.name = "pytorch"
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=pt_backend):
            with patch("ml_switcheroo_compiler.distributed.host_collectives._dispatch_pytorch_collective", return_value=None):
                res_pt_none = all_reduce(inp, op="SUM")
                np.testing.assert_allclose(res_pt_none, inp)

        # JAX backend mock
        jax_backend = MagicMock()
        jax_backend.name = "jax"
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=jax_backend):
            res = all_reduce(inp, op="SUM")
            assert res is not None
            assert all_gather(inp, axis=0) is not None
            assert broadcast(inp, root=0) is not None
    finally:
        config.eager_mode = old_eager


def test_dispatch_hardware_collective_edge_cases() -> None:
    """Verify hardware collective driver error and unavailability edge cases."""
    inp = np.array([1.0, 2.0], dtype=np.float32)

    with patch("ml_switcheroo_compiler.backends.cuda.nccl_collectives.NCCLDriver.is_available", return_value=False):
        assert _dispatch_hardware_collective("cuda", inp) is None

    with patch("ml_switcheroo_compiler.backends.cuda.nccl_collectives.NCCLDriver", side_effect=RuntimeError("nccl error")):
        assert _dispatch_hardware_collective("cuda", inp) is None

    with patch("ml_switcheroo_compiler.backends.rocm.rccl_collectives.RCCLDriver.is_available", return_value=False):
        assert _dispatch_hardware_collective("rocm", inp) is None

    with patch("ml_switcheroo_compiler.backends.rocm.rccl_collectives.RCCLDriver", side_effect=RuntimeError("rccl error")):
        assert _dispatch_hardware_collective("rocm", inp) is None


def test_dispatch_eager_collective_fallbacks_and_errors() -> None:
    """Verify fallbacks and exception handling in collective dispatching."""
    inp = np.array([1.0, 2.0], dtype=np.float32)

    # PyTorch and JAX exception handling and unknown op fallback
    with patch("ml_switcheroo_compiler.backends.pytorch.distributed_collectives.pytorch_all_reduce", side_effect=RuntimeError("pt fail")):
        assert _dispatch_pytorch_collective("AllReduce", inp) is None

    assert _dispatch_pytorch_collective("UnknownOp", inp) is None
    assert _dispatch_jax_collective("UnknownOp", inp) is None

    # Unhandled collective op in _invoke_driver_op fallback branch
    from ml_switcheroo_compiler.distributed.host_collectives import (
        _emulate_collective_ground_truth,
        _invoke_driver_op,
    )

    mock_driver = MagicMock()
    _invoke_driver_op(
        mock_driver,
        "UnknownCollectiveOp",
        (10, 20),
        (100, 100),
        0,
        0,
        None,
        None,
        world_size=1,
    )
    assert mock_driver.all_reduce.called

    # _emulate_collective_ground_truth with ranks_seq is None
    ground_none = _emulate_collective_ground_truth("AllReduce", inp, None, 1, 0)
    np.testing.assert_allclose(ground_none, inp)

    # _emulate_collective_ground_truth with Broadcast
    ground_bc = _emulate_collective_ground_truth("Broadcast", inp, [inp, inp * 2], 2, 0, root=1)
    np.testing.assert_allclose(ground_bc, inp * 2)

    # _emulate_collective_ground_truth with unknown op (fall-through branch returning copied tensor)
    ground_unknown = _emulate_collective_ground_truth("UnknownOp123", inp, [inp, inp * 2], 2, 0)
    np.testing.assert_allclose(ground_unknown, inp)

    # dispatch_eager_collective when get_active_backend() raises an exception (falls back to backend_name = 'numpy')
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=RuntimeError("no active backend")):
        res_backend_err = all_reduce(inp, op="SUM")
        assert res_backend_err is not None

    # test unhandled backend in dispatch_eager_collective (e.g. backend_name not matching torch, jax, etc.)
    unknown_backend = MagicMock()
    unknown_backend.name = "custom_unknown_backend"
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=unknown_backend):
        res_custom = all_reduce(inp, op="SUM")
        assert res_custom is not None

    with patch("ml_switcheroo_compiler.backends.jax.distributed_collectives.jax_all_reduce", side_effect=RuntimeError("jax fail")):
        assert _dispatch_jax_collective("AllReduce", inp) is None

    # NumPy unknown op fallback
    unknown_res = _dispatch_numpy_collective("UnknownOp", inp)
    np.testing.assert_allclose(unknown_res, inp)

    # Backend resolution exception fallback to numpy
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=RuntimeError("no active backend")):
        res = _dispatch_eager_collective("AllReduce", inp)
        np.testing.assert_allclose(res, inp)

    # Fallback to numpy when pytorch/jax returns None
    pt_mock = MagicMock()
    pt_mock.name = "pytorch"
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=pt_mock):
        with patch("ml_switcheroo_compiler.distributed.host_collectives._dispatch_pytorch_collective", return_value=None):
            res_pt = _dispatch_eager_collective("AllReduce", inp)
            np.testing.assert_allclose(res_pt, inp)

    jax_mock = MagicMock()
    jax_mock.name = "jax"
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=jax_mock):
        with patch("ml_switcheroo_compiler.distributed.host_collectives._dispatch_jax_collective", return_value=None):
            res_jax = _dispatch_eager_collective("AllReduce", inp)
            np.testing.assert_allclose(res_jax, inp)


def test_host_collective_communicator_edge_cases() -> None:
    """Verify HostCollectiveCommunicator operations with all_ranks_data=None and various reduction ops."""
    comm = HostCollectiveCommunicator(world_size=1, rank=0)
    data = np.array([2.0, 3.0], dtype=np.float32)

    # When all_ranks_data is None
    assert np.allclose(comm.all_reduce(data, all_ranks_data=None), data)
    assert np.allclose(comm.all_gather(data, all_ranks_data=None), data)
    assert np.allclose(comm.broadcast(data, all_ranks_data=None), data)
    assert np.allclose(comm.all_to_all(data, all_ranks_data=None), data)

    # Different reduction operators
    ranks = [np.array([2.0, 10.0]), np.array([4.0, 5.0])]
    comm2 = HostCollectiveCommunicator(world_size=2, rank=0)
    assert np.allclose(comm2.all_reduce(data, op="MAX", all_ranks_data=ranks), np.array([4.0, 10.0]))
    assert np.allclose(comm2.all_reduce(data, op="MIN", all_ranks_data=ranks), np.array([2.0, 5.0]))
    assert np.allclose(comm2.all_reduce(data, op="PROD", all_ranks_data=ranks), np.array([8.0, 50.0]))
    assert np.allclose(comm2.all_reduce(data, op="OTHER", all_ranks_data=ranks), np.array([6.0, 15.0]))


def test_accelerator_collective_nccl_full_mathematical_parity() -> None:
    """Verify accelerator collective execution matches mathematical expectations with buffer handling."""
    from ml_switcheroo_compiler.backends.cuda.nccl_collectives import NCCLDriver
    from ml_switcheroo_compiler.distributed.host_collectives import _dispatch_accelerator_collective

    driver = NCCLDriver()
    driver.nccl_lib = MagicMock()

    with patch("ml_switcheroo_compiler.backends.cuda.nccl_collectives.NCCLDriver", return_value=driver):
        arr1 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        arr2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ranks = [arr1, arr2]

        # AllReduce SUM
        res_sum = _dispatch_accelerator_collective("cuda", arr1, op_name="AllReduce", op="SUM", all_ranks_data=ranks)
        assert res_sum is not None
        np.testing.assert_allclose(res_sum, [11.0, 22.0, 33.0, 44.0])

        # AllReduce PROD
        res_prod = _dispatch_accelerator_collective("cuda", arr1, op_name="AllReduce", op="PROD", all_ranks_data=ranks)
        assert res_prod is not None
        np.testing.assert_allclose(res_prod, [10.0, 40.0, 90.0, 160.0])

        # AllGather
        res_ag = _dispatch_accelerator_collective("cuda", arr1, op_name="AllGather", axis=0, all_ranks_data=ranks)
        assert res_ag is not None
        assert res_ag.shape == (8,)
        np.testing.assert_allclose(res_ag, np.concatenate(ranks, axis=0))

        # ReduceScatter
        res_rs = _dispatch_accelerator_collective("cuda", arr1, op_name="ReduceScatter", op="SUM", scatter_dim=0, rank=0, all_ranks_data=ranks)
        assert res_rs is not None
        assert res_rs.shape == (2,)
        np.testing.assert_allclose(res_rs, [11.0, 22.0])

        # Broadcast
        res_bc = _dispatch_accelerator_collective("cuda", arr1, op_name="Broadcast", root=1, all_ranks_data=ranks)
        assert res_bc is not None
        np.testing.assert_allclose(res_bc, arr2)

        # AllToAll
        arr2d_1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        arr2d_2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        res_a2a = _dispatch_accelerator_collective("cuda", arr2d_1, op_name="AllToAll", scatter_dim=0, gather_dim=0, rank=0, all_ranks_data=[arr2d_1, arr2d_2])
        assert res_a2a is not None


def test_accelerator_collective_rccl_full_mathematical_parity() -> None:
    """Verify RCCL accelerator collective execution matches mathematical expectations."""
    from ml_switcheroo_compiler.backends.rocm.rccl_collectives import RCCLDriver
    from ml_switcheroo_compiler.distributed.host_collectives import _dispatch_accelerator_collective

    driver = RCCLDriver()
    driver.rccl_lib = MagicMock()

    with patch("ml_switcheroo_compiler.backends.rocm.rccl_collectives.RCCLDriver", return_value=driver):
        arr1 = np.array([5.0, 15.0], dtype=np.float64)
        arr2 = np.array([2.0, 8.0], dtype=np.float64)
        ranks = [arr1, arr2]

        res_max = _dispatch_accelerator_collective("rocm", arr1, op_name="AllReduce", op="MAX", all_ranks_data=ranks)
        assert res_max is not None
        np.testing.assert_allclose(res_max, [5.0, 15.0])

        res_min = _dispatch_accelerator_collective("rocm", arr1, op_name="AllReduce", op="MIN", all_ranks_data=ranks)
        assert res_min is not None
        np.testing.assert_allclose(res_min, [2.0, 8.0])


def test_nccl_and_rccl_driver_buffer_methods() -> None:
    """Verify buffer allocation, copy, and stream synchronization on NCCL and RCCL drivers."""
    from ml_switcheroo_compiler.backends.cuda.nccl_collectives import NCCLDriver
    from ml_switcheroo_compiler.backends.rocm.rccl_collectives import RCCLDriver

    for driver_cls in (NCCLDriver, RCCLDriver):
        driver = driver_cls()
        size_bytes = 64
        ptr = driver.allocate_buffer(size_bytes)
        assert ptr > 0

        src_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        driver.copy_host_to_device(src_arr, ptr)

        dst_arr = np.empty_like(src_arr)
        driver.copy_device_to_host(ptr, dst_arr)
        np.testing.assert_allclose(dst_arr, src_arr)

        assert driver.stream_synchronize(None) == 0
        driver.free_buffer(ptr)
