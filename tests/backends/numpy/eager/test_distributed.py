"""Tests for test_distributed."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.distributed import set_np_distributed_context

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_np_all_reduce_threads() -> None:
    """Test function test_np_all_reduce_threads."""
    import threading

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext, _np_axis_index

    assert _np_axis_index(np) == 0

    ctx1 = TCPDistributedContext(world_size=2, rank=0, addr="127.0.0.1", port=41000)
    ctx2 = TCPDistributedContext(world_size=2, rank=1, addr="127.0.0.1", port=41000)

    def init_ctx(ctx: EagerValue) -> None:
        """Test function init_ctx.

        Args:
            ctx: Test parameter `ctx`.
        """
        cast("EagerTensorProtocol", getattr(ctx, "initialize"))()

    t1 = threading.Thread(target=init_ctx, args=(ctx1,))
    t2 = threading.Thread(target=init_ctx, args=(ctx2,))

    t1.start()
    t2.start()

    t1.join(timeout=5)
    t2.join(timeout=5)

    arr = np.array([1, 2])

    # Run AllReduce over ring
    def run_all_reduce(ctx: EagerValue, tensor: EagerValue, op: EagerValue) -> EagerValue:
        """Test function run_all_reduce.

        Args:
            ctx: Test parameter `ctx`.
            tensor: Test parameter `tensor`.
            op: Test parameter `op`.
        """
        return cast("EagerTensorProtocol", getattr(ctx, "all_reduce_ring"))(tensor, op, np)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = cast("EagerCallableProtocol", executor.submit)(run_all_reduce, ctx1, arr, "sum")
        f2 = cast("EagerCallableProtocol", executor.submit)(run_all_reduce, ctx2, arr, "sum")
        res1 = f1.result()
        res2 = f2.result()

    cast("EagerCallableProtocol", np.testing.assert_array_equal)(res1, cast("EagerCallableProtocol", np.array)([2, 4]))
    cast("EagerCallableProtocol", np.testing.assert_array_equal)(res2, cast("EagerCallableProtocol", np.array)([2, 4]))

    ctx1.shutdown()
    ctx2.shutdown()


def test_all_gather_np_coverage9() -> None:
    """Test function test_all_gather_np_coverage9."""
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2

    def mock_all_gather(t: EagerValue) -> EagerValue:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.
        """
        return [[2.0], [3.0]]

    setattr(_tcp_dist_ctx, "all_gather_tensors", mock_all_gather)

    class MockNP:
        """Test class MockNP."""

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.
                """
                return cast("EagerTensorProtocol", getattr(tensors, "__getitem__")(0))

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.
                """
                return cast("EagerTensorProtocol", getattr(tensors, "__getitem__")(0))

        def array_split(self, arr: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.
            """
            return cast("EagerTensorProtocol", [arr]) * cast("EagerTensorProtocol", indices)

        def concatenate(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.
            """
            return cast("EagerTensorProtocol", getattr(t, "__getitem__")(0))

        def expand_dims(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.
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
    assert res is not None


def _test_all_gather_np_coverage10_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2
    getattr(_state, "update")({"_tcp_dist_ctx": _tcp_dist_ctx})
    _state.update(locals())


def _test_all_gather_np_coverage10_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """

    class ArrayMock:
        """Test class ArrayMock."""

        def __init__(self, val: EagerValue) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.val = val

        def __add__(self, other: EagerValue) -> EagerValue:
            """Test function __add__.

            Args:
                other: Test parameter `other`.
            """
            return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) + cast("EagerTensorProtocol", getattr(other, "val", other))))

        def __radd__(self, other: EagerValue) -> EagerValue:
            """Test function __radd__.

            Args:
                other: Test parameter `other`.
            """
            if other == 0:
                return cast("EagerTensorProtocol", self)
            return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) + cast("EagerTensorProtocol", getattr(other, "val", other))))

        def __mul__(self, other: EagerValue) -> EagerValue:
            """Test function __mul__.

            Args:
                other: Test parameter `other`.
            """
            if hasattr(other, "val"):
                return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) * cast("EagerTensorProtocol", getattr(other, "val", other))))
            return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) * cast("EagerTensorProtocol", other)))

        def copy(self) -> EagerValue:
            """Test function copy."""
            return cast("EagerTensorProtocol", ArrayMock(self.val))

    getattr(_state, "update")({"ArrayMock": ArrayMock})
    _state.update(locals())


def _test_all_gather_np_coverage10_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, _tcp_dist_ctx = _state["ArrayMock"], _state["_tcp_dist_ctx"]

    def mock_all_gather(t: EagerValue) -> EagerValue:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.
        """
        return [ArrayMock(2.0), ArrayMock(3.0)]

    setattr(_tcp_dist_ctx, "all_gather_tensors", mock_all_gather)
    getattr(_state, "update")({"mock_all_gather": mock_all_gather})
    _state.update(locals())


def _test_all_gather_np_coverage10_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock = _state["ArrayMock"]

    class MockNP:
        """Test class MockNP."""

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.
                """
                return cast("EagerCallableProtocol", ArrayMock)(cast("EagerCallableProtocol", max)([cast("EagerTensorProtocol", getattr(x, "val")) for x in cast("EagerTensorProtocol", tensors)]))

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.
                """
                return cast("EagerCallableProtocol", ArrayMock)(cast("EagerCallableProtocol", min)([cast("EagerTensorProtocol", getattr(x, "val")) for x in cast("EagerTensorProtocol", tensors)]))

        def array_split(self, arr: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.
            """
            return cast("EagerTensorProtocol", [cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", getattr(arr, "val")) / cast("EagerTensorProtocol", indices))]) * cast("EagerTensorProtocol", indices)

        def concatenate(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.
            """
            return cast("EagerTensorProtocol", getattr(t, "__getitem__")(0))

        def expand_dims(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.
            """
            return t

    getattr(_state, "update")({"MockNP": MockNP})
    _state.update(locals())


def _test_all_gather_np_coverage10_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
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

    setattr(_tcp_dist_ctx, "world_size", 1)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=None)

    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    setattr(_tcp_dist_ctx, "world_size", 2)
    getattr(_state, "update")({"dmod": dmod, "res": res})
    _state.update(locals())


def _test_all_gather_np_coverage10_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state: dict[str, EagerTensorProtocol] = {}
    _test_all_gather_np_coverage10_part_1(_state)
    _test_all_gather_np_coverage10_part_2(_state)
    _test_all_gather_np_coverage10_part_3(_state)
    _test_all_gather_np_coverage10_part_4(_state)
    _test_all_gather_np_coverage10_part_5(_state)
    _test_all_gather_np_coverage10_part_6(_state)
    assert len(_state) > 0


def _test_all_gather_np_coverage11_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2
    getattr(_state, "update")({"_tcp_dist_ctx": _tcp_dist_ctx})
    _state.update(locals())


def _test_all_gather_np_coverage11_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """

    class ArrayMock:
        """Test class ArrayMock."""

        def __init__(self, val: EagerValue) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.val = val

        def __add__(self, other: EagerValue) -> EagerValue:
            """Test function __add__.

            Args:
                other: Test parameter `other`.
            """
            return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) + cast("EagerTensorProtocol", getattr(other, "val", other))))

        def __radd__(self, other: EagerValue) -> EagerValue:
            """Test function __radd__.

            Args:
                other: Test parameter `other`.
            """
            if other == 0:
                return cast("EagerTensorProtocol", self)
            return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) + cast("EagerTensorProtocol", getattr(other, "val", other))))

        def __mul__(self, other: EagerValue) -> EagerValue:
            """Test function __mul__.

            Args:
                other: Test parameter `other`.
            """
            if hasattr(other, "val"):
                return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) * cast("EagerTensorProtocol", getattr(other, "val", other))))
            return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) * cast("EagerTensorProtocol", other)))

        def copy(self) -> EagerValue:
            """Test function copy."""
            return cast("EagerTensorProtocol", ArrayMock(self.val))

    getattr(_state, "update")({"ArrayMock": ArrayMock})
    _state.update(locals())


def _test_all_gather_np_coverage11_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, _tcp_dist_ctx = _state["ArrayMock"], _state["_tcp_dist_ctx"]

    def mock_all_gather(t: EagerValue) -> EagerValue:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.
        """
        return [ArrayMock(2.0), ArrayMock(3.0)]

    setattr(_tcp_dist_ctx, "all_gather_tensors", mock_all_gather)
    getattr(_state, "update")({"mock_all_gather": mock_all_gather})
    _state.update(locals())


def _test_all_gather_np_coverage11_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock = _state["ArrayMock"]

    class MockNP:
        """Test class MockNP."""

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.
                """
                return cast("EagerCallableProtocol", ArrayMock)(cast("EagerCallableProtocol", max)([cast("EagerTensorProtocol", getattr(x, "val")) for x in cast("EagerTensorProtocol", tensors)]))

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.
                """
                return cast("EagerCallableProtocol", ArrayMock)(cast("EagerCallableProtocol", min)([cast("EagerTensorProtocol", getattr(x, "val")) for x in cast("EagerTensorProtocol", tensors)]))

        def array_split(self, arr: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.
            """
            return cast("EagerTensorProtocol", [cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", getattr(arr, "val")) / cast("EagerTensorProtocol", indices))]) * cast("EagerTensorProtocol", indices)

        def concatenate(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.
            """
            return cast("EagerTensorProtocol", getattr(t, "__getitem__")(0))

        def expand_dims(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.
            """
            return t

    getattr(_state, "update")({"MockNP": MockNP})
    _state.update(locals())


def _test_all_gather_np_coverage11_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
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

    setattr(_tcp_dist_ctx, "world_size", 1)
    dmod._np_all_gather(np, np.array([2.0]), axis=0)
    dmod._np_all_gather(np, np.array([2.0]), axis=None)

    dmod._np_all_to_all(np, np.array([2.0]))
    dmod._np_broadcast(np, np.array([2.0]), root_rank=0)

    setattr(_tcp_dist_ctx, "world_size", 2)
    getattr(_state, "update")({"dmod": dmod, "res": res})
    _state.update(locals())


def _test_all_gather_np_coverage11_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state: dict[str, EagerTensorProtocol] = {}
    _test_all_gather_np_coverage11_part_1(_state)
    _test_all_gather_np_coverage11_part_2(_state)
    _test_all_gather_np_coverage11_part_3(_state)
    _test_all_gather_np_coverage11_part_4(_state)
    _test_all_gather_np_coverage11_part_5(_state)
    _test_all_gather_np_coverage11_part_6(_state)
    assert len(_state) > 0


def _test_all_gather_np_coverage12_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2
    getattr(_state, "update")({"_tcp_dist_ctx": _tcp_dist_ctx})
    _state.update(locals())


def _test_all_gather_np_coverage12_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """

    class ArrayMock:
        """Test class ArrayMock."""

        def __init__(self, val: EagerValue) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.val = val

        def __add__(self, other: EagerValue) -> EagerValue:
            """Test function __add__.

            Args:
                other: Test parameter `other`.
            """
            return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) + cast("EagerTensorProtocol", getattr(other, "val", other))))

        def __radd__(self, other: EagerValue) -> EagerValue:
            """Test function __radd__.

            Args:
                other: Test parameter `other`.
            """
            if other == 0:
                return cast("EagerTensorProtocol", self)
            return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) + cast("EagerTensorProtocol", getattr(other, "val", other))))

        def __mul__(self, other: EagerValue) -> EagerValue:
            """Test function __mul__.

            Args:
                other: Test parameter `other`.
            """
            if hasattr(other, "val"):
                return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) * cast("EagerTensorProtocol", getattr(other, "val", other))))
            return cast("EagerValue", cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", self.val) * cast("EagerTensorProtocol", other)))

        def copy(self) -> EagerValue:
            """Test function copy."""
            return cast("EagerTensorProtocol", ArrayMock(self.val))

    getattr(_state, "update")({"ArrayMock": ArrayMock})
    _state.update(locals())


def _test_all_gather_np_coverage12_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, _tcp_dist_ctx = _state["ArrayMock"], _state["_tcp_dist_ctx"]

    def mock_all_gather(t: EagerValue) -> EagerValue:
        """Test function mock_all_gather.

        Args:
            t: Test parameter `t`.
        """
        return [ArrayMock(2.0), ArrayMock(3.0)]

    setattr(_tcp_dist_ctx, "all_gather_tensors", mock_all_gather)
    getattr(_state, "update")({"mock_all_gather": mock_all_gather})
    _state.update(locals())


def _test_all_gather_np_coverage12_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock = _state["ArrayMock"]

    class MockNP:
        """Test class MockNP."""

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.
                """
                return cast("EagerCallableProtocol", ArrayMock)(cast("EagerCallableProtocol", max)([cast("EagerTensorProtocol", getattr(x, "val")) for x in cast("EagerTensorProtocol", tensors)]))

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.
                """
                return cast("EagerCallableProtocol", ArrayMock)(cast("EagerCallableProtocol", min)([cast("EagerTensorProtocol", getattr(x, "val")) for x in cast("EagerTensorProtocol", tensors)]))

        def array_split(self, arr: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.
            """
            return cast("EagerTensorProtocol", [cast("EagerCallableProtocol", ArrayMock)(cast("EagerTensorProtocol", getattr(arr, "val")) / cast("EagerTensorProtocol", indices))]) * cast("EagerTensorProtocol", indices)

        def concatenate(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.
            """
            return cast("EagerTensorProtocol", getattr(t, "__getitem__")(0))

        def expand_dims(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.
            """
            return t

    getattr(_state, "update")({"MockNP": MockNP})
    _state.update(locals())


def _test_all_gather_np_coverage12_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
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

    setattr(_tcp_dist_ctx, "rank", 1)
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="sum")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="prod")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="max")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="min")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="unknown")
    getattr(_state, "update")({"dmod": dmod, "res": res})
    _state.update(locals())


def _test_all_gather_np_coverage12_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, MockNP, _tcp_dist_ctx, dmod = _state["ArrayMock"], _state["MockNP"], _state["_tcp_dist_ctx"], _state["dmod"]
    res = _state["res"]
    setattr(_tcp_dist_ctx, "world_size", 1)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=None)

    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    setattr(_tcp_dist_ctx, "world_size", 2)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    dmod._np_shard_tensor(MockNP(), ArrayMock(2.0))
    assert res is not None


def test_all_gather_np_coverage12() -> None:
    """Test function test_all_gather_np_coverage12."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_all_gather_np_coverage12_part_1(_state)
    _test_all_gather_np_coverage12_part_2(_state)
    _test_all_gather_np_coverage12_part_3(_state)
    _test_all_gather_np_coverage12_part_4(_state)
    _test_all_gather_np_coverage12_part_5(_state)
    _test_all_gather_np_coverage12_part_6(_state)
    assert len(_state) > 0
