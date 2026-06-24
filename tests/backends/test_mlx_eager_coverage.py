from unittest.mock import MagicMock, patch

import pytest


def test_mlx_eager_coverage_part1():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not installed")

    from ml_switcheroo_compiler.backends.mlx.eager import execute_op

    class DummyBackend:
        pass

    cls = DummyBackend()

    res = execute_op(cls, "TakeAlongAxis", mx.array([1, 2]), mx.array([0]), axis=0)
    assert res is not None

    res = execute_op(cls, "Take", mx.array([1, 2]), mx.array([0]), axis=0)
    assert res is not None

    with patch(
        "ml_switcheroo_compiler.backends.numpy.eager.execute_op", return_value="scatter_res"
    ):
        execute_op(cls, "ScatterNd", mx.array([1]), mx.array([0]), mx.array([1]))  # dummy args

    # Reshape coverage
    shape_mock = MagicMock()
    shape_mock.data = [2]
    shape_mock.tolist.return_value = [2]

    res = execute_op(cls, "Reshape", mx.array([1, 2]), shape=shape_mock)
    assert res is not None

    res = execute_op(cls, "Reshape", mx.array([1, 2]), newshape=(2,))
    assert res is not None


def test_mlx_eager_coverage_part2():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX not installed")

    from ml_switcheroo_compiler.backends.mlx.eager import execute_op

    class DummyBackend:
        pass

    cls = DummyBackend()
    shape_mock = MagicMock()
    shape_mock.data = [2]
    shape_mock.tolist.return_value = [2]

    # Zeros
    res = execute_op(cls, "Zeros", 2)
    assert res is not None

    res = execute_op(cls, "Zeros", shape=shape_mock, dtype="float32")
    assert res is not None

    # Ones
    res = execute_op(cls, "Ones", 2)
    assert res is not None

    res = execute_op(cls, "Ones", shape=shape_mock, dtype="float32")
    assert res is not None

    # Full
    res = execute_op(cls, "Full", 2, 5)
    assert res is not None

    res = execute_op(cls, "Full", shape=shape_mock, fill_value=5, dtype="float32")
    assert res is not None

    # Trigger fallback without dtype
    res = execute_op(cls, "Zeros", shape=shape_mock, dtype=None)
    res = execute_op(cls, "Ones", shape=shape_mock, dtype=None)
    res = execute_op(cls, "Full", shape=shape_mock, fill_value=5, dtype=None)

    # Trigger type error in ones/zeros/full creation
    with patch("mlx.core.zeros", side_effect=[TypeError, "mock_res"]):
        res = execute_op(cls, "Zeros", shape=(2,), dtype="float32")
        assert res == "mock_res"

    with patch("mlx.core.ones", side_effect=[TypeError, "mock_res"]):
        res = execute_op(cls, "Ones", shape=(2,), dtype="float32")
        assert res == "mock_res"

    with patch("mlx.core.full", side_effect=[TypeError, "mock_res"]):
        res = execute_op(cls, "Full", shape=(2,), fill_value=5, dtype="float32")
        assert res == "mock_res"

    # Eye
    n_mock = MagicMock()
    n_mock.data = 2
    res = execute_op(cls, "Eye", n_mock, dtype="float32")
    assert res is not None

    # Exception fallback (NotImplementedError) -> route to numpy
    with patch("mlx.core.take", side_effect=NotImplementedError):
        res = execute_op(cls, "Take", mx.array([1, 2]), mx.array([0]), axis=0)
        assert getattr(res, "__class__", None).__name__ == "array"

    # Tuple return
    with patch("mlx.core.take", side_effect=NotImplementedError):
        with patch("ml_switcheroo_compiler.backends.numpy.eager.execute_op", return_value=(1, 2)):
            res = execute_op(cls, "Take", mx.array([1, 2]), mx.array([0]), axis=0)
            assert isinstance(res, tuple)
