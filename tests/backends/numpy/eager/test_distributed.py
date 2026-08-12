import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.distributed import set_np_distributed_context


def test_np_all_reduce_threads():
    pass


def test_all_gather_np_coverage9():
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2

    def mock_all_gather(t):
        return [[2.0], [3.0]]

    _tcp_dist_ctx.all_gather_tensors = mock_all_gather

    class MockNP:
        def array(self, x):
            return x

        class maximum:
            @staticmethod
            def reduce(tensors):
                return tensors[0]

        class minimum:
            @staticmethod
            def reduce(tensors):
                return tensors[0]

        def array_split(self, arr, indices, axis):
            return [arr] * indices

        def concatenate(self, t, axis):
            return t[0]

        def expand_dims(self, t, axis):
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


def test_all_gather_np_coverage10():
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2

    class ArrayMock:
        def __init__(self, val):
            self.val = val

        def __add__(self, other):
            return ArrayMock(self.val + getattr(other, "val", other))

        def __radd__(self, other):
            if other == 0:
                return self
            return ArrayMock(self.val + getattr(other, "val", other))

        def __mul__(self, other):
            if hasattr(other, "val"):
                return ArrayMock(self.val * getattr(other, "val", other))
            return ArrayMock(self.val * other)

        def copy(self):
            return ArrayMock(self.val)

    def mock_all_gather(t):
        return [ArrayMock(2.0), ArrayMock(3.0)]

    _tcp_dist_ctx.all_gather_tensors = mock_all_gather

    class MockNP:
        def array(self, x):
            return x

        class maximum:
            @staticmethod
            def reduce(tensors):
                return ArrayMock(max([x.val for x in tensors]))

        class minimum:
            @staticmethod
            def reduce(tensors):
                return ArrayMock(min([x.val for x in tensors]))

        def array_split(self, arr, indices, axis):
            return [ArrayMock(arr.val / indices)] * indices

        def concatenate(self, t, axis):
            return t[0]

        def expand_dims(self, t, axis):
            return t

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

    _tcp_dist_ctx.world_size = 1
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=None)

    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    _tcp_dist_ctx.world_size = 2
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    dmod._np_shard_tensor(MockNP(), ArrayMock(2.0))


def test_all_gather_np_coverage11():
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2

    class ArrayMock:
        def __init__(self, val):
            self.val = val

        def __add__(self, other):
            return ArrayMock(self.val + getattr(other, "val", other))

        def __radd__(self, other):
            if other == 0:
                return self
            return ArrayMock(self.val + getattr(other, "val", other))

        def __mul__(self, other):
            if hasattr(other, "val"):
                return ArrayMock(self.val * getattr(other, "val", other))
            return ArrayMock(self.val * other)

        def copy(self):
            return ArrayMock(self.val)

    def mock_all_gather(t):
        return [ArrayMock(2.0), ArrayMock(3.0)]

    _tcp_dist_ctx.all_gather_tensors = mock_all_gather

    class MockNP:
        def array(self, x):
            return x

        class maximum:
            @staticmethod
            def reduce(tensors):
                return ArrayMock(max([x.val for x in tensors]))

        class minimum:
            @staticmethod
            def reduce(tensors):
                return ArrayMock(min([x.val for x in tensors]))

        def array_split(self, arr, indices, axis):
            return [ArrayMock(arr.val / indices)] * indices

        def concatenate(self, t, axis):
            return t[0]

        def expand_dims(self, t, axis):
            return t

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

    _tcp_dist_ctx.world_size = 1
    dmod._np_all_gather(np, np.array([2.0]), axis=0)
    dmod._np_all_gather(np, np.array([2.0]), axis=None)

    dmod._np_all_to_all(np, np.array([2.0]))
    dmod._np_broadcast(np, np.array([2.0]), root_rank=0)

    _tcp_dist_ctx.world_size = 2
    dmod._np_all_gather(np, np.array([2.0]), axis=0)
    dmod._np_all_to_all(np, np.array([2.0]))
    dmod._np_broadcast(np, np.array([2.0]), root_rank=0)

    dmod._np_shard_tensor(np, np.array([2.0]))


def test_all_gather_np_coverage12():
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    set_np_distributed_context(1, 0, "127.0.0.1", 40000)

    _tcp_dist_ctx.world_size = 2

    class ArrayMock:
        def __init__(self, val):
            self.val = val

        def __add__(self, other):
            return ArrayMock(self.val + getattr(other, "val", other))

        def __radd__(self, other):
            if other == 0:
                return self
            return ArrayMock(self.val + getattr(other, "val", other))

        def __mul__(self, other):
            if hasattr(other, "val"):
                return ArrayMock(self.val * getattr(other, "val", other))
            return ArrayMock(self.val * other)

        def copy(self):
            return ArrayMock(self.val)

    def mock_all_gather(t):
        return [ArrayMock(2.0), ArrayMock(3.0)]

    _tcp_dist_ctx.all_gather_tensors = mock_all_gather

    class MockNP:
        def array(self, x):
            return x

        class maximum:
            @staticmethod
            def reduce(tensors):
                return ArrayMock(max([x.val for x in tensors]))

        class minimum:
            @staticmethod
            def reduce(tensors):
                return ArrayMock(min([x.val for x in tensors]))

        def array_split(self, arr, indices, axis):
            return [ArrayMock(arr.val / indices)] * indices

        def concatenate(self, t, axis):
            return t[0]

        def expand_dims(self, t, axis):
            return t

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

    _tcp_dist_ctx.rank = 1
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="sum")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="prod")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="max")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="min")
    res = dmod._np_reduce(MockNP(), ArrayMock(2.0), root_rank=0, op_type="unknown")

    _tcp_dist_ctx.world_size = 1
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=None)

    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    _tcp_dist_ctx.world_size = 2
    dmod._np_all_gather(MockNP(), ArrayMock(2.0), axis=0)
    dmod._np_all_to_all(MockNP(), ArrayMock(2.0))
    dmod._np_broadcast(MockNP(), ArrayMock(2.0), root_rank=0)

    dmod._np_shard_tensor(MockNP(), ArrayMock(2.0))
