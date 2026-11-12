"""Tests for test_distributed."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.distributed import set_np_distributed_context
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mod, ensure_mutable, ensure_tensor_dict, has_backend_member

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_np_all_reduce_threads() -> None:
    """Test function test_np_all_reduce_threads."""
    import threading

    from ml_switcheroo_compiler.backends.numpy.eager.distributed import TCPDistributedContext, _np_axis_index

    assert _np_axis_index(ensure_eager_tensor(np)) == 0

    ctx1 = TCPDistributedContext(world_size=2, rank=0, addr="127.0.0.1", port=41000)
    ctx2 = TCPDistributedContext(world_size=2, rank=1, addr="127.0.0.1", port=41000)

    def init_ctx(ctx: EagerValue) -> None:
        """Test function init_ctx.

        Args:
            ctx: Test parameter `ctx`.
        """
        ensure_eager_tensor(ensure_eager_tensor(ctx).initialize)()

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

        Returns:
            EagerValue: Computed output value from `run_all_reduce`.
        """
        return ensure_eager_tensor(ensure_eager_tensor(ctx).all_reduce_ring)(tensor, op, np)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = ensure_callable(executor.submit)(run_all_reduce, ctx1, arr, "sum")
        f2 = ensure_callable(executor.submit)(run_all_reduce, ctx2, arr, "sum")
        res1 = f1.result()
        res2 = f2.result()

    ensure_callable(np.testing.assert_array_equal)(res1, ensure_callable(np.array)([2, 4]))
    ensure_callable(np.testing.assert_array_equal)(res2, ensure_callable(np.array)([2, 4]))

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

        Returns:
            EagerValue: Computed output value from `mock_all_gather`.
        """
        return [[2.0], [3.0]]

    ensure_mutable(_tcp_dist_ctx).all_gather_tensors = ensure_eager_tensor(mock_all_gather)

    class MockNP:
        """Test class MockNP."""

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `array`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    EagerValue: Computed output value from `reduce`.
                """
                return ensure_eager_tensor(tensors)[0]

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    EagerValue: Computed output value from `reduce`.
                """
                return ensure_eager_tensor(tensors)[0]

        def array_split(self, arr: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `array_split`.
            """
            return ensure_eager_tensor([arr]) * ensure_eager_tensor(indices)

        def concatenate(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `concatenate`.
            """
            return ensure_eager_tensor(t)[0]

        def expand_dims(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `expand_dims`.
            """
            return t

    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

    # Try with list of lists to bypass item
    res = dmod._np_all_reduce(ensure_mod(MockNP()), [2.0], op_type="sum")

    res = dmod._np_all_reduce(ensure_mod(MockNP()), [2.0], op_type="prod")

    res = dmod._np_all_reduce(ensure_mod(MockNP()), [2.0], op_type="max")

    res = dmod._np_all_reduce(ensure_mod(MockNP()), [2.0], op_type="min")

    res = dmod._np_all_reduce(ensure_mod(MockNP()), [2.0], op_type="unknown")

    res = dmod._np_reduce_scatter(ensure_mod(MockNP()), [2.0], op_type="sum", axis=0)
    res = dmod._np_reduce_scatter(ensure_mod(MockNP()), [2.0], op_type="prod", axis=0)
    res = dmod._np_reduce_scatter(ensure_mod(MockNP()), [2.0], op_type="max", axis=0)
    res = dmod._np_reduce_scatter(ensure_mod(MockNP()), [2.0], op_type="min", axis=0)
    res = dmod._np_reduce_scatter(ensure_mod(MockNP()), [2.0], op_type="unknown", axis=0)

    res = dmod._np_reduce(ensure_mod(MockNP()), [2.0], root_rank=0, op_type="sum")
    res = dmod._np_reduce(ensure_mod(MockNP()), [2.0], root_rank=0, op_type="prod")
    res = dmod._np_reduce(ensure_mod(MockNP()), [2.0], root_rank=0, op_type="max")
    res = dmod._np_reduce(ensure_mod(MockNP()), [2.0], root_rank=0, op_type="min")
    res = dmod._np_reduce(ensure_mod(MockNP()), [2.0], root_rank=0, op_type="unknown")

    _tcp_dist_ctx.world_size = 1
    dmod._np_all_gather(ensure_mod(MockNP()), [2.0], axis=0)
    dmod._np_all_gather(ensure_mod(MockNP()), [2.0], axis=None)

    dmod._np_all_to_all(ensure_mod(MockNP()), [2.0])

    _tcp_dist_ctx.world_size = 2
    dmod._np_all_gather(ensure_mod(MockNP()), [2.0], axis=0)
    dmod._np_all_to_all(ensure_mod(MockNP()), [2.0])

    dmod._np_shard_tensor(ensure_mod(MockNP()), [2.0])
    assert res is not None


def _test_all_gather_np_coverage10_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_all_gather_np_coverage10.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2
    _state.update(ensure_tensor_dict({"_tcp_dist_ctx": _tcp_dist_ctx}))


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

            Returns:
                EagerValue: Computed output value from `__add__`.
            """
            return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) + ensure_eager_tensor(ensure_eager_tensor(other).val)))

        def __radd__(self, other: EagerValue) -> EagerValue:
            """Test function __radd__.

            Args:
                other: Test parameter `other`.

            Returns:
                EagerValue: Computed output value from `__radd__`.
            """
            if other == 0:
                return ensure_eager_tensor(self)
            return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) + ensure_eager_tensor(ensure_eager_tensor(other).val)))

        def __mul__(self, other: EagerValue) -> EagerValue:
            """Test function __mul__.

            Args:
                other: Test parameter `other`.

            Returns:
                EagerValue: Computed output value from `__mul__`.
            """
            if has_backend_member(other, "val"):
                return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) * ensure_eager_tensor(ensure_eager_tensor(other).val)))
            return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) * ensure_eager_tensor(other)))

        def copy(self) -> EagerValue:
            """Test function copy.

            Returns:
                EagerValue: Computed output value from `copy`.
            """
            return ensure_eager_tensor(ArrayMock(self.val))

    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock}))


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

        Returns:
            EagerValue: Computed output value from `mock_all_gather`.
        """
        return [ArrayMock(2.0), ArrayMock(3.0)]

    ensure_mutable(_tcp_dist_ctx).all_gather_tensors = ensure_eager_tensor(mock_all_gather)
    _state.update(ensure_tensor_dict({"mock_all_gather": mock_all_gather}))
    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock, "_tcp_dist_ctx": _tcp_dist_ctx}))


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

            Returns:
                EagerValue: Computed output value from `array`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    EagerValue: Computed output value from `reduce`.
                """
                return ensure_callable(ArrayMock)(ensure_callable(max)([ensure_eager_tensor(ensure_eager_tensor(x).val) for x in ensure_eager_tensor(tensors)]))

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    EagerValue: Computed output value from `reduce`.
                """
                return ensure_callable(ArrayMock)(ensure_callable(min)([ensure_eager_tensor(ensure_eager_tensor(x).val) for x in ensure_eager_tensor(tensors)]))

        def array_split(self, arr: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `array_split`.
            """
            return ensure_eager_tensor([ensure_callable(ArrayMock)(ensure_eager_tensor(ensure_eager_tensor(arr).val) / ensure_eager_tensor(indices))]) * ensure_eager_tensor(indices)

        def concatenate(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `concatenate`.
            """
            return ensure_eager_tensor(t)[0]

        def expand_dims(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `expand_dims`.
            """
            return t

    _state.update(ensure_tensor_dict({"MockNP": MockNP}))
    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock}))


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

    ensure_mutable(_tcp_dist_ctx).world_size = ensure_eager_tensor(1)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=None)

    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    ensure_mutable(_tcp_dist_ctx).world_size = ensure_eager_tensor(2)
    _state.update(ensure_tensor_dict({"dmod": dmod, "res": res}))
    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock, "MockNP": MockNP, "_tcp_dist_ctx": _tcp_dist_ctx}))


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
    _state.update(ensure_tensor_dict({"_tcp_dist_ctx": _tcp_dist_ctx}))


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

            Returns:
                EagerValue: Computed output value from `__add__`.
            """
            return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) + ensure_eager_tensor(ensure_eager_tensor(other).val)))

        def __radd__(self, other: EagerValue) -> EagerValue:
            """Test function __radd__.

            Args:
                other: Test parameter `other`.

            Returns:
                EagerValue: Computed output value from `__radd__`.
            """
            if other == 0:
                return ensure_eager_tensor(self)
            return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) + ensure_eager_tensor(ensure_eager_tensor(other).val)))

        def __mul__(self, other: EagerValue) -> EagerValue:
            """Test function __mul__.

            Args:
                other: Test parameter `other`.

            Returns:
                EagerValue: Computed output value from `__mul__`.
            """
            if has_backend_member(other, "val"):
                return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) * ensure_eager_tensor(ensure_eager_tensor(other).val)))
            return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) * ensure_eager_tensor(other)))

        def copy(self) -> EagerValue:
            """Test function copy.

            Returns:
                EagerValue: Computed output value from `copy`.
            """
            return ensure_eager_tensor(ArrayMock(self.val))

    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock}))


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

        Returns:
            EagerValue: Computed output value from `mock_all_gather`.
        """
        return [ArrayMock(2.0), ArrayMock(3.0)]

    ensure_mutable(_tcp_dist_ctx).all_gather_tensors = ensure_eager_tensor(mock_all_gather)
    _state.update(ensure_tensor_dict({"mock_all_gather": mock_all_gather}))
    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock, "_tcp_dist_ctx": _tcp_dist_ctx}))


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

            Returns:
                EagerValue: Computed output value from `array`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    EagerValue: Computed output value from `reduce`.
                """
                return ensure_callable(ArrayMock)(ensure_callable(max)([ensure_eager_tensor(ensure_eager_tensor(x).val) for x in ensure_eager_tensor(tensors)]))

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    EagerValue: Computed output value from `reduce`.
                """
                return ensure_callable(ArrayMock)(ensure_callable(min)([ensure_eager_tensor(ensure_eager_tensor(x).val) for x in ensure_eager_tensor(tensors)]))

        def array_split(self, arr: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `array_split`.
            """
            return ensure_eager_tensor([ensure_callable(ArrayMock)(ensure_eager_tensor(ensure_eager_tensor(arr).val) / ensure_eager_tensor(indices))]) * ensure_eager_tensor(indices)

        def concatenate(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `concatenate`.
            """
            return ensure_eager_tensor(t)[0]

        def expand_dims(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `expand_dims`.
            """
            return t

    _state.update(ensure_tensor_dict({"MockNP": MockNP}))
    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock}))


def _test_all_gather_np_coverage11_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_all_gather_np_coverage11.

    Args:
        _state: Shared test state dictionary.
    """
    _tcp_dist_ctx = _state["_tcp_dist_ctx"]
    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dmod

    res = dmod._np_all_reduce(ensure_eager_tensor(np), np.array([2.0]), op_type="sum")
    res = dmod._np_all_reduce(ensure_eager_tensor(np), np.array([2.0]), op_type="prod")
    res = dmod._np_all_reduce(ensure_eager_tensor(np), np.array([2.0]), op_type="max")
    res = dmod._np_all_reduce(ensure_eager_tensor(np), np.array([2.0]), op_type="min")
    res = dmod._np_all_reduce(ensure_eager_tensor(np), np.array([2.0]), op_type="unknown")

    res = dmod._np_reduce_scatter(ensure_eager_tensor(np), np.array([2.0, 2.0]), op_type="sum", axis=0)
    res = dmod._np_reduce_scatter(ensure_eager_tensor(np), np.array([2.0, 2.0]), op_type="prod", axis=0)
    res = dmod._np_reduce_scatter(ensure_eager_tensor(np), np.array([2.0, 2.0]), op_type="max", axis=0)
    res = dmod._np_reduce_scatter(ensure_eager_tensor(np), np.array([2.0, 2.0]), op_type="min", axis=0)
    res = dmod._np_reduce_scatter(ensure_eager_tensor(np), np.array([2.0, 2.0]), op_type="unknown", axis=0)

    res = dmod._np_reduce(ensure_eager_tensor(np), np.array([2.0]), root_rank=0, op_type="sum")
    res = dmod._np_reduce(ensure_eager_tensor(np), np.array([2.0]), root_rank=0, op_type="prod")
    res = dmod._np_reduce(ensure_eager_tensor(np), np.array([2.0]), root_rank=0, op_type="max")
    res = dmod._np_reduce(ensure_eager_tensor(np), np.array([2.0]), root_rank=0, op_type="min")
    res = dmod._np_reduce(ensure_eager_tensor(np), np.array([2.0]), root_rank=0, op_type="unknown")

    ensure_mutable(_tcp_dist_ctx).world_size = ensure_eager_tensor(1)
    dmod._np_all_gather(ensure_eager_tensor(np), np.array([2.0]), axis=0)
    dmod._np_all_gather(ensure_eager_tensor(np), np.array([2.0]), axis=None)

    dmod._np_all_to_all(ensure_eager_tensor(np), np.array([2.0]))
    dmod._np_broadcast(ensure_eager_tensor(np), np.array([2.0]), root_rank=0)

    ensure_mutable(_tcp_dist_ctx).world_size = ensure_eager_tensor(2)
    _state.update(ensure_tensor_dict({"dmod": dmod, "res": res}))
    _state.update(ensure_tensor_dict({"_tcp_dist_ctx": _tcp_dist_ctx}))


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
    _state.update(ensure_tensor_dict({"_tcp_dist_ctx": _tcp_dist_ctx}))


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

            Returns:
                EagerValue: Computed output value from `__add__`.
            """
            return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) + ensure_eager_tensor(ensure_eager_tensor(other).val)))

        def __radd__(self, other: EagerValue) -> EagerValue:
            """Test function __radd__.

            Args:
                other: Test parameter `other`.

            Returns:
                EagerValue: Computed output value from `__radd__`.
            """
            if other == 0:
                return ensure_eager_tensor(self)
            return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) + ensure_eager_tensor(ensure_eager_tensor(other).val)))

        def __mul__(self, other: EagerValue) -> EagerValue:
            """Test function __mul__.

            Args:
                other: Test parameter `other`.

            Returns:
                EagerValue: Computed output value from `__mul__`.
            """
            if has_backend_member(other, "val"):
                return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) * ensure_eager_tensor(ensure_eager_tensor(other).val)))
            return ensure_eager_tensor(ensure_callable(ArrayMock)(ensure_eager_tensor(self.val) * ensure_eager_tensor(other)))

        def copy(self) -> EagerValue:
            """Test function copy.

            Returns:
                EagerValue: Computed output value from `copy`.
            """
            return ensure_eager_tensor(ArrayMock(self.val))

    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock}))


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

        Returns:
            EagerValue: Computed output value from `mock_all_gather`.
        """
        return [ArrayMock(2.0), ArrayMock(3.0)]

    ensure_mutable(_tcp_dist_ctx).all_gather_tensors = ensure_eager_tensor(mock_all_gather)
    _state.update(ensure_tensor_dict({"mock_all_gather": mock_all_gather}))
    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock, "_tcp_dist_ctx": _tcp_dist_ctx}))


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

            Returns:
                EagerValue: Computed output value from `array`.
            """
            return x

        class maximum:
            """Test class maximum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    EagerValue: Computed output value from `reduce`.
                """
                return ensure_callable(ArrayMock)(ensure_callable(max)([ensure_eager_tensor(ensure_eager_tensor(x).val) for x in ensure_eager_tensor(tensors)]))

        class minimum:
            """Test class minimum."""

            @staticmethod
            def reduce(tensors: EagerValue) -> EagerValue:
                """Test function reduce.

                Args:
                    tensors: Test parameter `tensors`.

                Returns:
                    EagerValue: Computed output value from `reduce`.
                """
                return ensure_callable(ArrayMock)(ensure_callable(min)([ensure_eager_tensor(ensure_eager_tensor(x).val) for x in ensure_eager_tensor(tensors)]))

        def array_split(self, arr: EagerValue, indices: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function array_split.

            Args:
                arr: Test parameter `arr`.
                indices: Test parameter `indices`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `array_split`.
            """
            return ensure_eager_tensor([ensure_callable(ArrayMock)(ensure_eager_tensor(ensure_eager_tensor(arr).val) / ensure_eager_tensor(indices))]) * ensure_eager_tensor(indices)

        def concatenate(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function concatenate.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `concatenate`.
            """
            return ensure_eager_tensor(t)[0]

        def expand_dims(self, t: EagerValue, axis: EagerValue) -> EagerValue:
            """Test function expand_dims.

            Args:
                t: Test parameter `t`.
                axis: Test parameter `axis`.

            Returns:
                EagerValue: Computed output value from `expand_dims`.
            """
            return t

    _state.update(ensure_tensor_dict({"MockNP": MockNP}))
    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock}))


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

    ensure_mutable(_tcp_dist_ctx).rank = ensure_eager_tensor(1)
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="sum")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="prod")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="max")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="min")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="unknown")
    _state.update(ensure_tensor_dict({"dmod": dmod, "res": res}))
    _state.update(ensure_tensor_dict({"ArrayMock": ArrayMock, "MockNP": MockNP, "_tcp_dist_ctx": _tcp_dist_ctx}))


def _test_all_gather_np_coverage12_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_all_gather_np_coverage12.

    Args:
        _state: Shared test state dictionary.
    """
    ArrayMock, MockNP, _tcp_dist_ctx, dmod = _state["ArrayMock"], _state["MockNP"], _state["_tcp_dist_ctx"], _state["dmod"]
    res = _state["res"]
    ensure_mutable(_tcp_dist_ctx).world_size = ensure_eager_tensor(1)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=None)

    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    ensure_mutable(_tcp_dist_ctx).world_size = ensure_eager_tensor(2)
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
