import numpy as np
from ml_switcheroo_compiler.backends.numpy.eager.shape import (
    _mvlgamma,
    _np_dynamic_partition,
    _np_dynamic_stitch,
    _np_tensor_scatter_sub,
    _np_extract_volume_patches,
    _np_boolean_mask,
    _np_unravel_index,
    _np_argsort,
    _np_argwhere,
    _np_argpartition,
    _np_assign_add,
    _np_assign_sub,
)


def test_numpy_shape_eager_extra():
    # mvlgamma
    res = _mvlgamma(np.array([2.0, 3.0]), 2)

    # dynamic partition
    data = np.array([10, 20, 30, 40, 50])
    partitions = np.array([0, 0, 1, 1, 0])
    res = _np_dynamic_partition(np, data, partitions, num_partitions=2)
    assert len(res) == 2

    # dynamic stitch
    indices = [np.array([0, 4]), np.array([1, 2, 3])]
    data_list = [np.array([10, 50]), np.array([20, 30, 40])]
    res = _np_dynamic_stitch(np, indices, data_list)
    assert np.allclose(res, [10, 20, 30, 40, 50])

    # tensor scatter sub
    tensor = np.ones((5,))
    indices = np.array([[1], [3]])
    updates = np.array([0.5, 0.5])
    res = _np_tensor_scatter_sub(np, tensor, indices, updates)
    assert res[1] == 0.5

    # extract volume patches
    volume = np.ones((1, 4, 4, 4, 1))
    try:
        res = _np_extract_volume_patches(np, volume, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], "VALID")
    except NotImplementedError:
        pass

    # boolean mask
    tensor = np.array([1, 2, 3])
    mask = np.array([True, False, True])
    res = _np_boolean_mask(np, tensor, mask)
    assert len(res) == 2

    # unravel index
    res = _np_unravel_index(np, np.array([22, 41, 37]), (7, 6))

    # argsort, argwhere, argpartition
    _np_argsort(np, np.array([3, 1, 2]), axis=-1)
    _np_argwhere(np, np.array([True, False, True]))
    _np_argpartition(np, np.array([3, 4, 2, 1]), 1)

    # assign_add / assign_sub (mock arrays)
    ref = np.array([1, 2, 3])
    val = np.array([1, 1, 1])
    _np_assign_add(np, ref, val)
    _np_assign_sub(np, ref, val)


def test_numpy_shape_eager_extra2():
    from ml_switcheroo_compiler.backends.numpy.eager.shape import _np_boolean_mask, _np_broadcast_to
    import numpy as np

    tensor = np.array([[1, 2], [3, 4]])
    mask = np.array([True, False])
    _np_boolean_mask(np, tensor, mask, axis=0)

    # broadcast to branches
    # kwargs.get("shape") vs args[0]
    _np_broadcast_to(np, np.ones(2), (2, 2))
    _np_broadcast_to(np, np.ones(2), shape=(2, 2))


def test_numpy_shape_eager_extra3():
    from ml_switcheroo_compiler.backends.numpy.eager.shape import (
        _np_broadcast_in_dim,
        _np_tensor_scatter_sub,
        _np_argsort,
        _np_argpartition,
    )
    import numpy as np

    # broadcast in dim wrapper
    # mock global registry
    import ml_switcheroo_compiler.backends.eager_registry as reg_mod

    original_get = reg_mod.global_eager_registry.get

    def mock_get(name):
        return lambda backend, *args, **kwargs: args[0]

    reg_mod.global_eager_registry.get = mock_get
    try:
        _np_broadcast_in_dim(np, np.ones(2))
    finally:
        reg_mod.global_eager_registry.get = original_get

    # tensor scatter sub
    tensor = np.ones((5, 5))
    indices = np.array([[[1, 1], [3, 3]]])
    updates = np.array([[0.5, 0.5]])
    _np_tensor_scatter_sub(np, tensor, indices, updates)

    # argsort, argpartition kwargs
    _np_argsort(np, np.array([3, 1, 2]), dimension=None)
    _np_argpartition(np, np.array([3, 4, 2, 1]), 1, axis=None)


def test_numpy_shape_eager_extra4():
    from ml_switcheroo_compiler.backends.numpy.eager.shape import _np_dynamic_stitch
    import numpy as np

    indices = [np.array([], dtype=np.int32)]
    data_list = [np.array([])]
    _np_dynamic_stitch(np, indices, data_list)
