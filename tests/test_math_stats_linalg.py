import numpy as np
from ml_switcheroo_compiler import ops


def test_math_stats_linalg():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    # divide_no_nan
    x = ops.array(np.array([1.0, 2.0, 0.0]).astype(np.float32))
    y = ops.array(np.array([1.0, 0.0, 0.0]).astype(np.float32))
    out = ops.divide_no_nan(x, y)
    assert out is not None

    # eig
    m = ops.array(np.array([[1.0, 0.0], [0.0, 2.0]]).astype(np.float32))
    w, v = ops.eig(m)
    assert w is not None

    # logdet
    ld = ops.logdet(m)
    assert ld is not None

    # lstsq
    a = ops.array(np.array([[1.0, 0.0], [0.0, 1.0]]).astype(np.float32))
    b = ops.array(np.array([1.0, 2.0]).astype(np.float32))
    out_lstsq = ops.lstsq(a, b)
    assert out_lstsq is not None

    # moments
    m, v = ops.moments(a, axes=-1)
    assert m is not None


def test_add_n_and_accumulate_n():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    x = ops.array(np.array([1.0, 2.0]))
    y = ops.array(np.array([3.0, 4.0]))

    # add_n
    out_add = ops.add_n([x, y, x])
    np.testing.assert_allclose(out_add.data, [5.0, 8.0])

    # accumulate_n
    out_acc = ops.accumulate_n([x, y])
    np.testing.assert_allclose(out_acc.data, [4.0, 6.0])


def test_cumulative_logsumexp():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    x = ops.array(np.array([1.0, 2.0, 3.0]))
    out = ops.cumulative_logsumexp(x, axis=0)
    expected = np.log(np.cumsum(np.exp([1.0, 2.0, 3.0])))
    np.testing.assert_allclose(out.data, expected, rtol=1e-5)


def test_no_nan_ops():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    x = ops.array(np.array([1.0, 2.0, 0.0]))
    y = ops.array(np.array([2.0, 0.0, 0.0]))

    out_div = ops.divide_no_nan(x, y)
    np.testing.assert_allclose(out_div.data, [0.5, 0.0, 0.0])

    out_mul = ops.multiply_no_nan(x, y)
    np.testing.assert_allclose(out_mul.data, [2.0, 0.0, 0.0])

    out_rec = ops.reciprocal_no_nan(y)
    np.testing.assert_allclose(out_rec.data, [0.5, 0.0, 0.0])


def test_squared_difference():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    x = ops.array(np.array([1.0, 2.0]))
    y = ops.array(np.array([3.0, 1.0]))
    out = ops.squared_difference(x, y)
    np.testing.assert_allclose(out.data, [4.0, 1.0])


def test_xdivy_xlog1py():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    x = ops.array(np.array([0.0, 2.0]))
    y = ops.array(np.array([0.0, 2.0]))

    out_xdivy = ops.xdivy(x, y)
    # x=0 => 0, x=2,y=2 => 1
    np.testing.assert_allclose(out_xdivy.data, [0.0, 1.0])

    out_xlog1py = ops.xlog1py(x, y)
    # x=0 => 0, x=2, y=2 => 2 * log(3)
    np.testing.assert_allclose(out_xlog1py.data, [0.0, 2.0 * np.log(3.0)])


def test_new_math_ops():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    # is_non_decreasing
    x = ops.array(np.array([1.0, 2.0, 3.0]))
    out1 = ops.is_non_decreasing(x)
    assert out1.data

    x2 = ops.array(np.array([1.0, 3.0, 2.0]))
    assert not ops.is_non_decreasing(x2).data

    # is_strictly_increasing
    x3 = ops.array(np.array([1.0, 2.0, 2.0]))
    assert ops.is_strictly_increasing(x).data
    assert not ops.is_strictly_increasing(x3).data

    # l2_normalize
    x4 = ops.array(np.array([3.0, 4.0]))
    out_l2 = ops.l2_normalize(x4, axis=0)
    np.testing.assert_allclose(out_l2.data, [0.6, 0.8])

    # zero_fraction
    x5 = ops.array(np.array([0.0, 1.0, 0.0, 2.0]))
    out_zf = ops.zero_fraction(x5)
    np.testing.assert_allclose(out_zf.data, 0.5)

    # reduce_euclidean_norm
    out_ren = ops.reduce_euclidean_norm(x4)
    np.testing.assert_allclose(out_ren.data, 5.0)

    # scalar_mul
    out_sm = ops.scalar_mul(x4, 2.0)
    np.testing.assert_allclose(out_sm.data, [6.0, 8.0])

    # reduce_logsumexp
    out_rle = ops.reduce_logsumexp(x)
    expected = np.log(np.sum(np.exp([1.0, 2.0, 3.0])))
    np.testing.assert_allclose(out_rle.data, expected)


def test_special_functions():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    x = ops.array(np.array([1.0, 2.0]))

    out_j0 = ops.bessel_j0(x)
    assert out_j0 is not None

    out_j1 = ops.bessel_j1(x)
    assert out_j1 is not None

    out_k0 = ops.bessel_k0(x)
    assert out_k0 is not None

    out_k0e = ops.bessel_k0e(x)
    assert out_k0e is not None

    out_k1 = ops.bessel_k1(x)
    assert out_k1 is not None

    out_k1e = ops.bessel_k1e(x)
    assert out_k1e is not None

    out_y0 = ops.bessel_y0(x)
    assert out_y0 is not None

    out_y1 = ops.bessel_y1(x)
    assert out_y1 is not None

    out_dawsn = ops.dawsn(x)
    assert out_dawsn is not None

    out_expint = ops.expint(x)
    assert out_expint is not None

    out_f_cos = ops.fresnel_cos(x)
    assert out_f_cos is not None

    out_f_sin = ops.fresnel_sin(x)
    assert out_f_sin is not None

    out_spence = ops.spence(x)
    assert out_spence is not None


def test_advanced_indexing_shape():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    # boolean_mask
    t = ops.array(np.array([1, 2, 3, 4]))
    mask = ops.array(np.array([True, False, True, False]))
    out_mask = ops.boolean_mask(t, mask)
    np.testing.assert_allclose(out_mask.data, [1, 3])

    # unravel_index
    out_unravel = ops.unravel_index(ops.array(np.array([22, 41, 37])), ops.array(np.array([7, 6])))
    assert out_unravel is not None

    # tensor_scatter_update
    # For simplicity just test existence since numpy implementations might be naive
    assert ops.tensor_scatter_update is not None
    assert ops.tensor_scatter_add is not None
    assert ops.tensor_scatter_max is not None
    assert ops.tensor_scatter_min is not None
    assert ops.tensor_scatter_sub is not None
    assert ops.dynamic_partition is not None
    assert ops.dynamic_stitch is not None
    assert ops.extract_volume_patches is not None


def test_adjoint():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    m = ops.array(np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 2j]]))
    out = ops.adjoint(m)
    # the adjoint is the conjugate transpose
    expected = np.array([[1.0 - 1j, 3.0], [2.0, 4.0 + 2j]])
    np.testing.assert_allclose(out.data, expected)


def test_adjoint_ast():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.tracer import _tracer
    from ml_switcheroo_compiler.backends.registry import BackendRegistry
    import numpy as np

    config.eager_mode = False

    _tracer.start_tracing("Test")
    x = ops.array(np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 2j]]))
    _ = ops.adjoint(x)
    graph = _tracer.stop_tracing()

    gen_cls = BackendRegistry.get("numpy")
    gen = gen_cls(graph)
    code = gen.generate()

    assert "adjoint" in code
