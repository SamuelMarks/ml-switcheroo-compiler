import pytest

from ml_switcheroo_compiler.backends.jax.eager import (
    _execute_accumulate_n,
    _execute_adaptive_avg_pool,
    _execute_bessel_jn,
    _execute_binom_cdf,
    _execute_cast,
    _execute_ragged_tensor_to_dense,
    _execute_unsorted_segment_max,
    _execute_unsorted_segment_min,
    _execute_unsorted_segment_prod,
    _execute_unsorted_segment_sum,
    _execute_variance,
    execute_op,
)


def test_jax_eager_functions():
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")

    # _execute_adaptive_avg_pool
    res = _execute_adaptive_avg_pool(jnp.ones((2, 2)), 1)
    assert res.shape == (2, 1)

    res2 = _execute_adaptive_avg_pool(jnp.ones((2, 2)), (1, 1))
    assert res2.shape == (1, 1)

    # _execute_accumulate_n
    res = _execute_accumulate_n([jnp.ones((2, 2)), jnp.ones((2, 2))])
    assert jnp.all(res == 2.0)

    # _execute_binom_cdf
    res = _execute_binom_cdf(1, 2, 0.5)
    assert res is not None

    # _execute_bessel_jn
    res = _execute_bessel_jn(1, 2.0)
    assert res is not None

    # _execute_unsorted_segment_*
    res = _execute_unsorted_segment_sum(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert res[0] == 3

    res = _execute_unsorted_segment_max(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert res[0] == 2

    res = _execute_unsorted_segment_min(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert res[0] == 1

    res = _execute_unsorted_segment_prod(jnp.array([1, 2, 3]), jnp.array([0, 0, 1]), 2)
    assert res[0] == 2

    # _execute_variance
    res = _execute_variance(jnp.array([1.0, 2.0, 3.0]))
    assert res is not None

    # _execute_cast
    class MockDtype:
        value = "int4"

    res = _execute_cast(jnp.array([1.0]), dtype=MockDtype())
    assert res.dtype == jnp.int8

    res = _execute_cast(jnp.array([1.0]), MockDtype())
    assert res.dtype == jnp.int8

    class MockDtype2:
        value = "bfloat16"

    res = _execute_cast(jnp.array([1.0]), MockDtype2())
    assert res.dtype == jnp.bfloat16

    class MockDtype3:
        value = "float16"

    res = _execute_cast(jnp.array([1.0]), MockDtype3())
    assert res.dtype == jnp.float16

    class MockDtype4:
        value = "float8"

    res = _execute_cast(jnp.array([1.0]), MockDtype4())
    assert res is not None

    class MockDtype5:
        value = "float32"

    res = _execute_cast(jnp.array([1.0]), MockDtype5())
    assert res.dtype == jnp.float32

    # _execute_ragged_tensor_to_dense
    res = _execute_ragged_tensor_to_dense([jnp.array([1.0]), jnp.array([1.0, 2.0])])
    assert res.shape == (2, 2)

    res = _execute_ragged_tensor_to_dense(jnp.array([1.0]))
    assert res.shape == (1,)

    # execute_op
    res = execute_op(None, "Add", jnp.array([1.0]), jnp.array([2.0]))
    assert res[0] == 3.0

    res = execute_op(None, "Variance", jnp.array([1.0, 2.0, 3.0]))
    assert res is not None

    # test some mapped ops
    res = execute_op(None, "Cumprod", jnp.array([1.0, 2.0, 3.0]))
    assert res is not None

    # test global_eager_registry dispatch
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    @global_eager_registry.register("DummyGlobalOp")
    def _dummy_global_op(jnp_module, *args, **kwargs):
        return jnp_module.array([42.0])

    res = execute_op(None, "DummyGlobalOp", jnp.array([1.0]))
    assert res[0] == 42.0

    # test backend not supported error
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with pytest.raises(BackendNotSupportedError):
        execute_op(None, "UnknownOp123")


def test_jax_eager_execute_op_lambdas():
    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.jax.eager import _OP_DISPATCH

    # Test lambda branches
    _OP_DISPATCH["ActivityRegularization"](1.0)
    _OP_DISPATCH["AdaptiveMaxPool3D_Indices"](jnp.ones((2, 2)), 1)
    _OP_DISPATCH["AdaptiveLogSoftmaxWithLoss"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["Adjoint"](jnp.ones((2, 2)))
    from unittest import mock

    with (
        mock.patch("jax.lax.all_gather", lambda x, **kw: x, create=True),
        mock.patch("jax.lax.all_to_all", lambda x, *args, **kw: x, create=True),
        mock.patch("jax.lax.psum", lambda x, **kw: x, create=True),
        mock.patch("jax.lax.pmax", lambda x, **kw: x, create=True),
        mock.patch("jax.lax.reduce_scatter", lambda x, *args, **kw: x, create=True),
    ):
        _OP_DISPATCH["AllGather"](jnp.ones((2,)))
        _OP_DISPATCH["AllToAll"](jnp.ones((2,)))
        _OP_DISPATCH["AllReduce"](jnp.ones((2,)), op_type="sum")
        _OP_DISPATCH["AllReduce"](jnp.ones((2,)), op_type="max")
        _OP_DISPATCH["ReduceScatter"](jnp.ones((2,)), op_type="sum")
        _OP_DISPATCH["ReduceScatter"](jnp.ones((2,)), op_type="max")

    _OP_DISPATCH["AlphaDropout"](jnp.ones((2,)))
    _OP_DISPATCH["AsString"](jnp.ones((2,)))
    _OP_DISPATCH["Assert"](True, jnp.ones((2,)))
    _OP_DISPATCH["Assign"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["AssignAdd"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["AssignSub"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["AssignVariable"](jnp.ones((2,)), jnp.ones((2,)))
    _OP_DISPATCH["AssociativeScan"](jnp.ones((2,)))
    _OP_DISPATCH["AssociativeScan"](lambda x: x, jnp.ones((2,)))

    try:
        _OP_DISPATCH["Frombuffer"](b"\x00\x00\x00\x00", dtype=jnp.float32)
    except Exception:
        pass

    _OP_DISPATCH["Fft2"](jnp.ones((2, 2)))
    _OP_DISPATCH["Fftfreq"](2)
    _OP_DISPATCH["Fftn"](jnp.ones((2, 2)))
    _OP_DISPATCH["Fftnd"](jnp.ones((2, 2)))
    _OP_DISPATCH["Fftshift"](jnp.ones((2, 2)))

    # HardSilu, HardSwish (if available in jax.nn)
    import jax.nn

    if hasattr(jax.nn, "hard_silu"):
        _OP_DISPATCH["HardSilu"](jnp.ones((2,)))
    if hasattr(jax.nn, "hard_swish"):
        _OP_DISPATCH["HardSwish"](jnp.ones((2,)))

    _OP_DISPATCH["Hfft"](jnp.ones((2,)))
    _OP_DISPATCH["Ifft"](jnp.ones((2,)))
    _OP_DISPATCH["Ifft2"](jnp.ones((2, 2)))
    _OP_DISPATCH["Ifftn"](jnp.ones((2, 2)))
    _OP_DISPATCH["Ifftnd"](jnp.ones((2, 2)))
    _OP_DISPATCH["Ifftshift"](jnp.ones((2, 2)))
    _OP_DISPATCH["Ihfft"](jnp.ones((2,)))
    _OP_DISPATCH["Irfft"](jnp.ones((2,)))
    _OP_DISPATCH["Irfft2"](jnp.ones((2, 2)))
    _OP_DISPATCH["Irfftn"](jnp.ones((2, 2)))
    _OP_DISPATCH["Irfftnd"](jnp.ones((2, 2)))
    with mock.patch("jax.nn.mish", create=True, new=None), mock.patch("jax.nn.squareplus", create=True, new=None):
        import jax.nn

        if hasattr(jax.nn, "mish"):
            del jax.nn.mish
        if hasattr(jax.nn, "squareplus"):
            del jax.nn.squareplus
        _OP_DISPATCH["Mish"](jnp.ones((2,)))
        _OP_DISPATCH["Squareplus"](jnp.ones((2,)))

    _OP_DISPATCH["LogSoftmax"](jnp.ones((2,)))
    _OP_DISPATCH["Mish"](jnp.ones((2,)))
    _OP_DISPATCH["OneHot"](jnp.array([0, 1]), 2)
    _OP_DISPATCH["OneHot"](jnp.array([0, 1]), depth=2)
    _OP_DISPATCH["Rfft"](jnp.ones((2,)))
    _OP_DISPATCH["Rfft2"](jnp.ones((2, 2)))
    _OP_DISPATCH["Rfftfreq"](2)
    _OP_DISPATCH["Rfftn"](jnp.ones((2, 2)))
    _OP_DISPATCH["Rfftnd"](jnp.ones((2, 2)))
    _OP_DISPATCH["Sigmoid"](jnp.ones((2,)))
    _OP_DISPATCH["Softmax"](jnp.ones((2,)))
    _OP_DISPATCH["Squareplus"](jnp.ones((2,)))


def test_jax_eager_execute_op_fallback():
    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.jax.eager import execute_op

    res = execute_op(None, "Mul", jnp.array([1.0]), jnp.array([2.0]))
    assert res[0] == 2.0

    res = execute_op(None, "Sub", jnp.array([1.0]), jnp.array([2.0]))
    assert res[0] == -1.0

    res = execute_op(None, "Div", jnp.array([1.0]), jnp.array([2.0]))
    assert res[0] == 0.5

    res = execute_op(None, "Eig", jnp.eye(2))
    assert res is not None


def test_jax_eager_extra():
    import pytest

    from ml_switcheroo_compiler.backends.jax.eager import _execute_accumulate_n, _execute_adaptive_avg_pool

    # AdaptivePool without shape
    # We must pass something that JAX can convert to an array but does not have .shape before conversion
    assert _execute_adaptive_avg_pool([1, 2], 1) is not None

    # AccumulateN empty
    with pytest.raises(ValueError):
        _execute_accumulate_n([])


def test_jax_adaptive_max_pool_tuple_size():
    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.jax.eager import _execute_adaptive_max_pool

    operand = jnp.ones((1, 1, 4, 4))
    res = _execute_adaptive_max_pool(operand, (2, 2))
    assert res.shape == (1, 1, 2, 2)
