from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import numpy_eager_registry


class BoomOps:
    def __getattr__(self, name):
        raise ValueError("Boom")


def test_math_misc_runtime_errors():
    import ml_switcheroo_compiler

    with patch.object(ml_switcheroo_compiler, "ops", BoomOps()):
        with pytest.raises(RuntimeError, match="Eager execution failed"):
            numpy_eager_registry._registry["RawMatMul"](None, np.array([[1]]), np.array([[2]]))

        with pytest.raises(RuntimeError, match="Eager execution failed"):
            numpy_eager_registry._registry["SparseDenseMatMul"](None, np.array([[1]]), np.array([[2]]))

        with pytest.raises(RuntimeError, match="Eager execution failed"):
            numpy_eager_registry._registry["rem"](None, np.array([5]), np.array([2]))

        with pytest.raises(RuntimeError, match="Eager execution failed"):
            numpy_eager_registry._registry["confusion_matrix"](None, np.array([0]), np.array([0]))

        with pytest.raises(RuntimeError, match="Eager execution failed"):
            numpy_eager_registry._registry["descriptive"](None, np.array([1]))

        with pytest.raises(RuntimeError, match="Eager execution failed"):
            numpy_eager_registry._registry["distributions"](None, np.array([1]))
