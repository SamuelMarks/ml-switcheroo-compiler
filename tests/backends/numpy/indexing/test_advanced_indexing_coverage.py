import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing import (
    _tensor_scatter_add,
    _tensor_scatter_max,
    _tensor_scatter_min,
    _tensor_scatter_update,
)


def test_tuple_instances() -> None:
    class MockIndex:
        def __array__(self):
            return np.array([[0, 0], [1, 1]])

    idx = MockIndex()
    updates = np.array([1, 4])
    t = np.zeros((2, 2))

    res = _tensor_scatter_update(t, idx, updates)
    assert res[0, 0] == 1

    res = _tensor_scatter_add(t, idx, updates)
    assert res[0, 0] == 1

    res = _tensor_scatter_max(t, idx, updates)
    assert res[0, 0] == 1

    res = _tensor_scatter_min(t, idx, updates)
    assert res[0, 0] == 0
