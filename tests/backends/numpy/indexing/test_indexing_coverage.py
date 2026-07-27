import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.indexing import _dynamic_update_slice


def test_dynamic_update_slice_with_item() -> None:
    class MockItem:
        # has item but not data
        def item(self):
            return 1

    x = np.zeros((3, 3))
    update = np.ones((2, 2))
    start_indices = [MockItem(), MockItem()]
    res = _dynamic_update_slice(x, update, start_indices)
    assert res[1, 1] == 1.0
