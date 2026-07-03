"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tree_util import tree_all, tree_leaves, tree_reduce, tree_structure


def test_tree_util_extra() -> object:
    """Function docstring."""
    device = Device("cpu")
    t1 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
    t2 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))
    #    t_false = Tensor(np.zeros((2,)), TensorConfig((2,), "float32", device))

    assert tree_leaves({"a": t1, "b": t2}) == [t1, t2]
    assert tree_structure({"a": t1, "b": t2}) is not None

    # test tree_all
    assert tree_all([True, True])
    assert not tree_all([True, False])

    # test tree_reduce
    assert tree_reduce(lambda x, y: x + y, [1, 2, 3]) == 6
    assert tree_reduce(lambda x, y: x + y, [1, 2, 3], 10) == 16
