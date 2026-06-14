"""Tests for extra random creation frontend ops."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.creation.frontend import manual_seed, rand, randint, randn
from ml_switcheroo_compiler.tracing.tracer import _tracer


def test_rand_frontend_eager() -> None:
    """Test function."""
    config.eager_mode = True
    t1 = rand(2, 3)
    assert t1.shape == (2, 3)
    t2 = randn(2, 3)
    assert t2.shape == (2, 3)
    t3 = randint(0, 10, (2, 3))
    assert t3.shape == (2, 3)
    assert manual_seed(42) == 42


def test_rand_frontend_tracing() -> None:
    """Test function."""
    config.eager_mode = False
    _tracer.start_tracing("test_rand")
    t1 = rand(2, 3)
    assert t1.shape == (2, 3)
    t2 = randn(2, 3)
    assert t2.shape == (2, 3)
    t3 = randint(0, 10, (2, 3))
    assert t3.shape == (2, 3)
    assert manual_seed(42) == 42

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.creation.frontend import array, diag

    arr = array([1, 2, 3], dtype=DType.Int32)
    assert arr.shape == (3,)

    # array without dtype
    array([1.0, 2.0])

    diag_out = diag(arr)
    assert diag_out.shape == (3, 3)

    arr_2d = array([[1, 2], [3, 4]])
    diag_out2 = diag(arr_2d)
    assert diag_out2.shape == (2,)

    import pytest

    with pytest.raises(ValueError):
        diag(array(1.0))

    _tracer.stop_tracing()

    # Trigger RuntimeError outside tracing
    with pytest.raises(RuntimeError):
        diag(arr)

    config.eager_mode = True
