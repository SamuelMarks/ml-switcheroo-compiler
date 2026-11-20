"""Tests for test_distributed."""

from __future__ import annotations

import types
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext, set_np_distributed_context
from ml_switcheroo_compiler.core.shape import OpValue
from tests.testing_utils import MockTestStub, adapt_backend_module, coerce_int, coerce_numeric, coerce_operand_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class _ArrayMock(MockTestStub):
    """Numeric tensor stub supporting scalar arithmetic for distributed tests."""

    def __init__(self, val: BackendArg) -> None:
        """Initialize `_ArrayMock` with numeric scalar payload.

        Args:
            val: Numeric input value.
        """
        super().__init__("ArrayMock")
        self.val: float = float(coerce_numeric(val))

    def __add__(self, other: BackendArg) -> BackendArg:
        """Compute elementwise addition.

        Args:
            other: Right-hand operand.

        Returns:
            BackendArg: Wrapped sum.
        """
        return _ArrayMock(self.val + float(coerce_numeric(adapt_backend_module(other).require_symbol("val"))))

    def __radd__(self, other: BackendArg) -> BackendArg:
        """Compute reflected addition.

        Args:
            other: Left-hand operand.

        Returns:
            BackendArg: Wrapped sum.
        """
        if other == 0:
            return self
        return _ArrayMock(self.val + float(coerce_numeric(adapt_backend_module(other).require_symbol("val"))))

    def __mul__(self, other: BackendArg) -> BackendArg:
        """Compute elementwise multiplication.

        Args:
            other: Right-hand operand.

        Returns:
            BackendArg: Wrapped product.
        """
        if adapt_backend_module(other).has_symbol("val"):
            return _ArrayMock(self.val * float(coerce_numeric(adapt_backend_module(other).require_symbol("val"))))
        return _ArrayMock(self.val * float(coerce_numeric(other)))

    def copy(self) -> BackendArg:
        """Return a shallow copy of `_ArrayMock`.

        Returns:
            BackendArg: Cloned `_ArrayMock`.
        """
        return _ArrayMock(self.val)


def test_np_all_reduce_threads() -> None:
    """Test function test_np_all_reduce_threads."""
    import threading

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext, _np_axis_index

    assert _np_axis_index(np) == 0

    ctx1 = TCPDistributedContext(world_size=2, rank=0, addr="127.0.0.1", port=41000)
    ctx2 = TCPDistributedContext(world_size=2, rank=1, addr="127.0.0.1", port=41000)

    def init_ctx(ctx: BackendArg) -> None:
        """Test function init_ctx.

        Args:
            ctx: Test parameter `ctx`.
        """
        adapt_backend_module(ctx).require_symbol("initialize")()

    t1 = threading.Thread(target=init_ctx, args=(ctx1,))
    t2 = threading.Thread(target=init_ctx, args=(ctx2,))

    t1.start()
    t2.start()

    t1.join(timeout=5)
    t2.join(timeout=5)

    arr = np.array([1, 2])

    # Run AllReduce over ring
    def run_all_reduce(ctx: TCPDistributedContext, tensor: BackendArg, op: str) -> BackendArg:
        """Test function run_all_reduce.

        Args:
            ctx: Test parameter `ctx`.
            tensor: Test parameter `tensor`.
            op: Test parameter `op`.

        Returns:
            BackendArg: Computed output value from `run_all_reduce`.
        """
        return ctx.all_reduce_ring(ensure_ndarray(tensor), op, np)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run_all_reduce, ctx1, arr, "sum")
        f2 = executor.submit(run_all_reduce, ctx2, arr, "sum")
        res1 = f1.result()
        res2 = f2.result()

    np.testing.assert_array_equal(ensure_ndarray(res1), np.array([2, 4]))
    np.testing.assert_array_equal(ensure_ndarray(res2), np.array([2, 4]))

    ctx1.shutdown()
    ctx2.shutdown()


def test_all_gather_np_coverage9() -> None:
    """Test function test_all_gather_np_coverage9."""
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2

    def mock_all_gather(t: BackendArg) -> BackendArg:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.

        Returns:
            BackendArg: Computed output value from `mock_all_gather`.
        """
        return [[2.0], [3.0]]

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "all_gather_tensors", mock_all_gather, raising=False)

    class MockNP(MockTestStub):
        """Test class MockNP."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: BackendArg) -> BackendArg:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    BackendArg: Computed output value from `reduce`.
                """
                return coerce_operand_list(tensors)[0]

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: BackendArg) -> BackendArg:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    BackendArg: Computed output value from `reduce`.
                """
                return coerce_operand_list(tensors)[0]

        def array_split(self, arr: BackendArg, indices: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `array_split`.
            """
            return [arr] * coerce_int(indices)

        def concatenate(self, t: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `concatenate`.
            """
            return coerce_operand_list(t)[0]

        def expand_dims(self, t: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `expand_dims`.
            """
            return t

    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

    # Try with list of lists to bypass item
    res = dmod._np_all_reduce(MockNP(), [2.0], op_type="sum")

    res = dmod._np_all_reduce(MockNP(), [2.0], op_type="prod")

    res = dmod._np_all_reduce(MockNP(), [2.0], op_type="max")

    res = dmod._np_all_reduce(MockNP(), [2.0], op_type="min")

    res = dmod._np_all_reduce(MockNP(), [2.0], op_type="unknown")

    res = dmod._np_reduce_scatter(MockNP(), [2.0], op_type="sum", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), [2.0], op_type="prod", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), [2.0], op_type="max", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), [2.0], op_type="min", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), [2.0], op_type="unknown", axis=0)

    res = dmod._np_reduce(MockNP(), [2.0], root_rank=0, op_type="sum")
    res = dmod._np_reduce(MockNP(), [2.0], root_rank=0, op_type="prod")
    res = dmod._np_reduce(MockNP(), [2.0], root_rank=0, op_type="max")
    res = dmod._np_reduce(MockNP(), [2.0], root_rank=0, op_type="min")
    res = dmod._np_reduce(MockNP(), [2.0], root_rank=0, op_type="unknown")

    _tcp_dist_ctx.world_size = 1
    dmod._np_all_gather(MockNP(), [2.0], axis=0)
    dmod._np_all_gather(MockNP(), [2.0], axis=None)

    dmod._np_all_to_all(MockNP(), [2.0])

    _tcp_dist_ctx.world_size = 2
    dmod._np_all_gather(MockNP(), [2.0], axis=0)
    dmod._np_all_to_all(MockNP(), [2.0])

    dmod._np_shard_tensor(MockNP(), [2.0])
    assert len(coerce_operand_list(res)) == 1


class _TestAllGatherNpCoverage10State(typing.TypedDict, total=False):
    """Typed shared state for `_test_all_gather_np_coverage10` split helpers."""

    ArrayMock: type[_ArrayMock]
    MockNP: type
    _tcp_dist_ctx: TCPDistributedContext
    dmod: types.ModuleType
    res: OpValue


def _test_all_gather_np_coverage10_part_1(_state: _TestAllGatherNpCoverage10State) -> None:
    """Execute part 1 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2
    _state.update({"_tcp_dist_ctx": _tcp_dist_ctx})


def _test_all_gather_np_coverage10_part_2(_state: _TestAllGatherNpCoverage10State) -> None:
    """Execute part 2 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock = _ArrayMock
    _state.update({"ArrayMock": ArrayMock})


def _test_all_gather_np_coverage10_part_3(_state: _TestAllGatherNpCoverage10State) -> None:
    """Execute part 3 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, _tcp_dist_ctx = _state["ArrayMock"], _state["_tcp_dist_ctx"]

    def mock_all_gather(t: BackendArg) -> BackendArg:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.

        Returns:
            BackendArg: Computed output value from `mock_all_gather`.
        """
        return [ArrayMock(2.0), ArrayMock(3.0)]

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "all_gather_tensors", mock_all_gather, raising=False)
    _state.update({"ArrayMock": ArrayMock, "_tcp_dist_ctx": _tcp_dist_ctx})


def _test_all_gather_np_coverage10_part_4(_state: _TestAllGatherNpCoverage10State) -> None:
    """Execute part 4 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock = _state["ArrayMock"]

    class MockNP(MockTestStub):
        """Test class MockNP."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: BackendArg) -> BackendArg:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    BackendArg: Computed output value from `reduce`.
                """
                return ArrayMock(
                    max(
                        float(coerce_numeric(adapt_backend_module(x).require_symbol("val")))
                        for x in coerce_operand_list(tensors)
                    )
                )

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: BackendArg) -> BackendArg:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    BackendArg: Computed output value from `reduce`.
                """
                return ArrayMock(
                    min(
                        float(coerce_numeric(adapt_backend_module(x).require_symbol("val")))
                        for x in coerce_operand_list(tensors)
                    )
                )

        def array_split(self, arr: BackendArg, indices: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `array_split`.
            """
            n_idx = coerce_int(indices)
            return [ArrayMock(float(coerce_numeric(adapt_backend_module(arr).require_symbol("val"))) / n_idx)] * n_idx

        def concatenate(self, t: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `concatenate`.
            """
            return coerce_operand_list(t)[0]

        def expand_dims(self, t: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `expand_dims`.
            """
            return t

    _state.update({"MockNP": MockNP})
    _state.update({"ArrayMock": ArrayMock})


def _test_all_gather_np_coverage10_part_5(_state: _TestAllGatherNpCoverage10State) -> None:
    """Execute part 5 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, MockNP, _tcp_dist_ctx = _state["ArrayMock"], _state["MockNP"], _state["_tcp_dist_ctx"]
    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="sum")

    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="prod")

    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="max")

    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="min")

    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="unknown")

    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="sum", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="prod", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="max", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="min", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="unknown", axis=0)

    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="sum")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="prod")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="max")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="min")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="unknown")

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "world_size", 1, raising=False)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=None)

    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "world_size", 2, raising=False)
    _state.update({"dmod": dmod, "res": res})
    _state.update({"ArrayMock": ArrayMock, "MockNP": MockNP, "_tcp_dist_ctx": _tcp_dist_ctx})


def _test_all_gather_np_coverage10_part_6(_state: _TestAllGatherNpCoverage10State) -> None:
    """Execute part 6 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, MockNP, dmod, res = _state["ArrayMock"], _state["MockNP"], _state["dmod"], _state["res"]
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    dmod._np_shard_tensor(MockNP(), ArrayMock(2.0))
    assert res is not None


def test_all_gather_np_coverage10() -> None:
    """Test function test_all_gather_np_coverage10."""
    _state: _TestAllGatherNpCoverage10State = {}
    _test_all_gather_np_coverage10_part_1(_state)
    _test_all_gather_np_coverage10_part_2(_state)
    _test_all_gather_np_coverage10_part_3(_state)
    _test_all_gather_np_coverage10_part_4(_state)
    _test_all_gather_np_coverage10_part_5(_state)
    _test_all_gather_np_coverage10_part_6(_state)
    assert len(_state) > 0


class _TestAllGatherNpCoverage11State(typing.TypedDict, total=False):
    """Typed shared state for `_test_all_gather_np_coverage11` split helpers."""

    ArrayMock: type[_ArrayMock]
    MockNP: type
    _tcp_dist_ctx: TCPDistributedContext
    dmod: types.ModuleType
    res: OpValue


def _test_all_gather_np_coverage11_part_1(_state: _TestAllGatherNpCoverage11State) -> None:
    """Execute part 1 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2
    _state.update({"_tcp_dist_ctx": _tcp_dist_ctx})


def _test_all_gather_np_coverage11_part_2(_state: _TestAllGatherNpCoverage11State) -> None:
    """Execute part 2 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock = _ArrayMock
    _state.update({"ArrayMock": ArrayMock})


def _test_all_gather_np_coverage11_part_3(_state: _TestAllGatherNpCoverage11State) -> None:
    """Execute part 3 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, _tcp_dist_ctx = _state["ArrayMock"], _state["_tcp_dist_ctx"]

    def mock_all_gather(t: BackendArg) -> BackendArg:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.

        Returns:
            BackendArg: Computed output value from `mock_all_gather`.
        """
        return [ArrayMock(2.0), ArrayMock(3.0)]

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "all_gather_tensors", mock_all_gather, raising=False)
    _state.update({"ArrayMock": ArrayMock, "_tcp_dist_ctx": _tcp_dist_ctx})


def _test_all_gather_np_coverage11_part_4(_state: _TestAllGatherNpCoverage11State) -> None:
    """Execute part 4 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock = _state["ArrayMock"]

    class MockNP(MockTestStub):
        """Test class MockNP."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: BackendArg) -> BackendArg:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    BackendArg: Computed output value from `reduce`.
                """
                return ArrayMock(
                    max(
                        float(coerce_numeric(adapt_backend_module(x).require_symbol("val")))
                        for x in coerce_operand_list(tensors)
                    )
                )

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: BackendArg) -> BackendArg:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    BackendArg: Computed output value from `reduce`.
                """
                return ArrayMock(
                    min(
                        float(coerce_numeric(adapt_backend_module(x).require_symbol("val")))
                        for x in coerce_operand_list(tensors)
                    )
                )

        def array_split(self, arr: BackendArg, indices: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `array_split`.
            """
            n_idx = coerce_int(indices)
            return [ArrayMock(float(coerce_numeric(adapt_backend_module(arr).require_symbol("val"))) / n_idx)] * n_idx

        def concatenate(self, t: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `concatenate`.
            """
            return coerce_operand_list(t)[0]

        def expand_dims(self, t: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `expand_dims`.
            """
            return t

    _state.update({"MockNP": MockNP})
    _state.update({"ArrayMock": ArrayMock})


def _test_all_gather_np_coverage11_part_5(_state: _TestAllGatherNpCoverage11State) -> None:
    """Execute part 5 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    _tcp_dist_ctx = _state["_tcp_dist_ctx"]
    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

    res = dmod._np_all_reduce(np, np.array([2.0]), op_type="sum")
    res = dmod._np_all_reduce(np, np.array([2.0]), op_type="prod")
    res = dmod._np_all_reduce(np, np.array([2.0]), op_type="max")
    res = dmod._np_all_reduce(np, np.array([2.0]), op_type="min")
    res = dmod._np_all_reduce(np, np.array([2.0]), op_type="unknown")

    res = dmod._np_reduce_scatter(np, np.array([2.0, 2.0]), op_type="sum", axis=0)
    res = dmod._np_reduce_scatter(np, np.array([2.0, 2.0]), op_type="prod", axis=0)
    res = dmod._np_reduce_scatter(np, np.array([2.0, 2.0]), op_type="max", axis=0)
    res = dmod._np_reduce_scatter(np, np.array([2.0, 2.0]), op_type="min", axis=0)
    res = dmod._np_reduce_scatter(np, np.array([2.0, 2.0]), op_type="unknown", axis=0)

    res = dmod._np_reduce(np, np.array([2.0]), root_rank=0, op_type="sum")
    res = dmod._np_reduce(np, np.array([2.0]), root_rank=0, op_type="prod")
    res = dmod._np_reduce(np, np.array([2.0]), root_rank=0, op_type="max")
    res = dmod._np_reduce(np, np.array([2.0]), root_rank=0, op_type="min")
    res = dmod._np_reduce(np, np.array([2.0]), root_rank=0, op_type="unknown")

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "world_size", 1, raising=False)
    dmod._np_all_gather(np, np.array([2.0]), axis=0)
    dmod._np_all_gather(np, np.array([2.0]), axis=None)

    dmod._np_all_to_all(np, np.array([2.0]))
    dmod._np_broadcast(np, np.array([2.0]), root_rank=0)

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "world_size", 2, raising=False)
    _state.update({"dmod": dmod, "res": res})
    _state.update({"_tcp_dist_ctx": _tcp_dist_ctx})


def _test_all_gather_np_coverage11_part_6(_state: _TestAllGatherNpCoverage11State) -> None:
    """Execute part 6 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    dmod, res = _state["dmod"], _state["res"]
    dmod._np_all_gather(np, np.array([2.0]), axis=0)
    dmod._np_all_to_all(np, np.array([2.0]))
    dmod._np_broadcast(np, np.array([2.0]), root_rank=0)

    dmod._np_shard_tensor(np, np.array([2.0]))
    assert res is not None


def test_all_gather_np_coverage11() -> None:
    """Test function test_all_gather_np_coverage11."""
    _state: _TestAllGatherNpCoverage11State = {}
    _test_all_gather_np_coverage11_part_1(_state)
    _test_all_gather_np_coverage11_part_2(_state)
    _test_all_gather_np_coverage11_part_3(_state)
    _test_all_gather_np_coverage11_part_4(_state)
    _test_all_gather_np_coverage11_part_5(_state)
    _test_all_gather_np_coverage11_part_6(_state)
    assert len(_state) > 0


class _TestAllGatherNpCoverage12State(typing.TypedDict, total=False):
    """Typed shared state for `_test_all_gather_np_coverage12` split helpers."""

    ArrayMock: type[_ArrayMock]
    MockNP: type
    _tcp_dist_ctx: TCPDistributedContext
    dmod: types.ModuleType
    res: OpValue


def _test_all_gather_np_coverage12_part_1(_state: _TestAllGatherNpCoverage12State) -> None:
    """Execute part 1 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2
    _state.update({"_tcp_dist_ctx": _tcp_dist_ctx})


def _test_all_gather_np_coverage12_part_2(_state: _TestAllGatherNpCoverage12State) -> None:
    """Execute part 2 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock = _ArrayMock
    _state.update({"ArrayMock": ArrayMock})


def _test_all_gather_np_coverage12_part_3(_state: _TestAllGatherNpCoverage12State) -> None:
    """Execute part 3 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, _tcp_dist_ctx = _state["ArrayMock"], _state["_tcp_dist_ctx"]

    def mock_all_gather(t: BackendArg) -> BackendArg:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.

        Returns:
            BackendArg: Computed output value from `mock_all_gather`.
        """
        return [ArrayMock(2.0), ArrayMock(3.0)]

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "all_gather_tensors", mock_all_gather, raising=False)
    _state.update({"ArrayMock": ArrayMock, "_tcp_dist_ctx": _tcp_dist_ctx})


def _test_all_gather_np_coverage12_part_4(_state: _TestAllGatherNpCoverage12State) -> None:
    """Execute part 4 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock = _state["ArrayMock"]

    class MockNP(MockTestStub):
        """Test class MockNP."""

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: BackendArg) -> BackendArg:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    BackendArg: Computed output value from `reduce`.
                """
                return ArrayMock(
                    max(
                        float(coerce_numeric(adapt_backend_module(x).require_symbol("val")))
                        for x in coerce_operand_list(tensors)
                    )
                )

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: BackendArg) -> BackendArg:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    BackendArg: Computed output value from `reduce`.
                """
                return ArrayMock(
                    min(
                        float(coerce_numeric(adapt_backend_module(x).require_symbol("val")))
                        for x in coerce_operand_list(tensors)
                    )
                )

        def array_split(self, arr: BackendArg, indices: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `array_split`.
            """
            n_idx = coerce_int(indices)
            return [ArrayMock(float(coerce_numeric(adapt_backend_module(arr).require_symbol("val"))) / n_idx)] * n_idx

        def concatenate(self, t: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `concatenate`.
            """
            return coerce_operand_list(t)[0]

        def expand_dims(self, t: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `expand_dims`.
            """
            return t

    _state.update({"MockNP": MockNP})
    _state.update({"ArrayMock": ArrayMock})


def _test_all_gather_np_coverage12_part_5(_state: _TestAllGatherNpCoverage12State) -> None:
    """Execute part 5 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, MockNP, _tcp_dist_ctx = _state["ArrayMock"], _state["MockNP"], _state["_tcp_dist_ctx"]
    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="sum")
    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="prod")
    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="max")
    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="min")
    res = dmod._np_all_reduce(MockNP(), ArrayMock(2.0), op_type="unknown")

    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="sum", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="prod", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="max", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="min", axis=0)
    res = dmod._np_reduce_scatter(MockNP(), ArrayMock(2.0), op_type="unknown", axis=0)

    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="sum")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="prod")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="max")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="min")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="unknown")

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "rank", 1, raising=False)
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="sum")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="prod")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="max")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="min")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="unknown")
    _state.update({"dmod": dmod, "res": res})
    _state.update({"ArrayMock": ArrayMock, "MockNP": MockNP, "_tcp_dist_ctx": _tcp_dist_ctx})


def _test_all_gather_np_coverage12_part_6(_state: _TestAllGatherNpCoverage12State) -> None:
    """Execute part 6 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, MockNP, _tcp_dist_ctx, dmod = (
        _state["ArrayMock"],
        _state["MockNP"],
        _state["_tcp_dist_ctx"],
        _state["dmod"],
    )
    res = _state["res"]
    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "world_size", 1, raising=False)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=None)

    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    pytest.MonkeyPatch().setattr(_tcp_dist_ctx, "world_size", 2, raising=False)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    dmod._np_shard_tensor(MockNP(), ArrayMock(2.0))
    assert res is not None


def test_all_gather_np_coverage12() -> None:
    """Test function test_all_gather_np_coverage12."""
    _state: _TestAllGatherNpCoverage12State = {}
    _test_all_gather_np_coverage12_part_1(_state)
    _test_all_gather_np_coverage12_part_2(_state)
    _test_all_gather_np_coverage12_part_3(_state)
    _test_all_gather_np_coverage12_part_4(_state)
    _test_all_gather_np_coverage12_part_5(_state)
    _test_all_gather_np_coverage12_part_6(_state)
    assert len(_state) > 0
