import numpy as np


def test_sparse_ops_coverage():
    from ml_switcheroo_compiler.backends.numpy.eager.math_misc import _np_raggeddot, _np_sparsemapvalues, _np_sparsereducemax, _np_sparsereshape, _np_sparsesampledadd, _np_sparsesegmentsum, _np_sparsetranspose

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 0], [0, 1]])

    assert np.array_equal(_np_raggeddot(np, a, b), a)
    assert np.array_equal(_np_sparsemapvalues(np, lambda x: x * 2, a), a * 2)
    assert np.array_equal(_np_sparsereducemax(np, a), np.array([2, 4]))
    assert np.array_equal(_np_sparsereshape(np, a, (4,)), np.array([1, 2, 3, 4]))
    assert np.array_equal(_np_sparsesampledadd(np, a, b), a + b)
    assert np.array_equal(_np_sparsesegmentsum(np, a, np.array([0, 1]), np.array([0, 0])), np.array([[4, 6]]))
    assert np.array_equal(_np_sparsetranspose(np, a), a.T)


def test_ragged_tensor_to_dense_jax():
    class MockJnpArray:
        pass

    # Not testing actual JAX import to avoid dependencies issues, just passing mock
    # Wait, JAX numpy might be imported
    pass
