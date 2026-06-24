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
