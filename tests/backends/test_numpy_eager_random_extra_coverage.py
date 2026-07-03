"""Module docstring."""

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.random as mod
from ml_switcheroo_compiler.backends.numpy.eager.random import _unwrap


def test_unwrap_coverage() -> object:
    """Function docstring."""
    # scalar ndarray
    arr = np.array(5)
    assert _unwrap(arr) == 5

    # 1D empty ndarray
    arr_empty = np.array([])
    assert _unwrap(arr_empty) == ()

    class DummyTensor:
        """Class docstring."""

        def __init__(self) -> object:
            """Function docstring."""
            self.data = np.array(10)

    class DummyTensorOuter:
        """Class docstring."""

        def __init__(self) -> object:
            """Function docstring."""
            self.__class__.__name__ = "Tensor"
            self.data = np.array(10)

    dt = DummyTensorOuter()
    assert _unwrap(dt) == 10


def test_randint_dtype() -> object:
    """Function docstring."""
    # pass size to make it an array
    res = mod._np_randint(np, 0, 10, size=(2,), dtype=np.int32)
    assert res.dtype == np.int32

    # custom dtype as string?
    class DummyDtype:
        """Class docstring."""

        def __str__(self) -> object:
            """Function docstring."""
            return "int32"

    res2 = mod._np_randint(np, 0, 10, size=(2,), dtype=DummyDtype())
    assert res2.dtype == np.int32


def test_dropout() -> object:
    """Function docstring."""
    res = mod.dropout(np, np.ones((2, 2)), 0.5)
    assert res is not None


def test_other_ops() -> object:
    """Function docstring."""
    for _, func in mod.numpy_eager_registry._registry.items():
        if func.__module__ == "ml_switcheroo_compiler.backends.numpy.eager.random":
            try:
                func(np, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
            except Exception:
                pass
            try:
                func(np, np.array([1.0, 2.0]))
            except Exception:
                pass
            try:
                func(np, np.array([1.0, 2.0]), 1.0)
            except Exception:
                pass


def test_seeds_and_shapes() -> object:
    """Function docstring."""
    mod._np_seed(np, 123)
    mod._np_manual_seed(np, 123)

    key = np.array([0, 0])
    mod._np_ball(np, key, 2, shape=(2,))
    mod._np_ball(np, key, 2, shape=2)
    mod._np_beta(np, key, 1.0, 1.0, shape=(2,))
    mod._np_binomial(np, key, 10, 0.5, shape=(2,))
    mod._np_dirichlet(np, key, np.array([1.0, 1.0]), shape=(2,))
    mod._np_dirichlet(np, key, np.array([1.0, 1.0]), shape=2)
    mod._np_ds_maxwell(np, key, 0.0, 1.0, shape=(2,))
    mod._np_f(np, key, 1.0, 1.0, shape=(2,))
    mod._np_multivariate_normal(np, key, np.array([0.0, 0.0]), np.eye(2), shape=(2,))
    mod._np_triangular(np, key, 0.0, 0.5, 1.0, shape=(2,))
    mod._np_wald(np, key, 1.0, 1.0, shape=(2,))
    mod._np_weibull(np, key, 1.0, 1.0, shape=(2,))
    mod._np_rng_uniform(np, 0.0, 1.0, shape=(2,))
