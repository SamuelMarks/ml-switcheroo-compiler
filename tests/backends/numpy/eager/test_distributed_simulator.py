import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager import numpy_eager_registry


def test_numpy_distributed_simulators():
    np_mod = np
    tensor = np.array([1, 2, 3])

    # AllGather
    gathered = numpy_eager_registry.get("AllGather")(np_mod, tensor, axis=0)
    assert gathered.shape == (1, 3)

    # AllReduce
    reduced = numpy_eager_registry.get("AllReduce")(np_mod, tensor)
    np.testing.assert_array_equal(reduced, tensor)

    # ReduceScatter
    scattered = numpy_eager_registry.get("ReduceScatter")(np_mod, tensor)
    np.testing.assert_array_equal(scattered, tensor)

    # AllToAll
    all_to_all = numpy_eager_registry.get("AllToAll")(np_mod, tensor)
    np.testing.assert_array_equal(all_to_all, tensor)
