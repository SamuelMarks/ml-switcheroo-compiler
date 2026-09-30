"""Tests for test_numpy_types_and_mixins."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.numpy import numpy_mixins as np_mixins
from ml_switcheroo_compiler.backends.numpy import types as np_types


class _MockBackendModule:
    """Mock backend module implementing zeros, array, and asarray."""

    def zeros(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """Mock zeros.

        Args:
            shape (tuple[int, ...]): Output shape.

        Returns:
            tuple[str, tuple[int, ...]]: Mock zeros descriptor.
        """
        return ("zeros", shape)

    def array(self, data: object, dtype: object = None) -> tuple[str, object, object]:
        """Mock array.

        Args:
            data (object): Input data.
            dtype (object): Optional data type.

        Returns:
            tuple[str, object, object]: Mock array descriptor.
        """
        return ("array", data, dtype)

    def asarray(self, data: object) -> tuple[str, object]:
        """Mock asarray.

        Args:
            data (object): Input data.

        Returns:
            tuple[str, object]: Mock asarray descriptor.
        """
        return ("asarray", data)


class _MockItemObject:
    """Mock object implementing item()."""

    def item(self) -> float:
        """Return scalar item value.

        Returns:
            float: Scalar float value.
        """
        return 42.5


class _MockDaskDtypeWrapper:
    """Mock dtype object with value attribute."""

    def __init__(self, val: str) -> None:
        """Initialize mock dtype wrapper.

        Args:
            val (str): Type name.
        """
        self.value = val


def _sample_add_fn(x: int, y: int = 10) -> int:
    """Add two integers.

    Args:
        x (int): First int.
        y (int): Second int.

    Returns:
        int: Sum of x and y.
    """
    return x + y


def _sample_shape_fn(a: int, b: int) -> tuple[int, int]:
    """Sample shape function returning tuple.

    Args:
        a (int): First dimension.
        b (int): Second dimension.

    Returns:
        tuple[int, int]: Pair (a, b).
    """
    return (a, b)


def test_backends_numpy_types_and_mixins() -> None:
    """Verify numpy backend types and mixins visitors."""

    class _DummyNumpyBackend:
        """Dummy backend class for numpy types."""

        pass

    z = np_types.zeros(_DummyNumpyBackend, (2, 2))
    assert z.shape == (2, 2)
    assert z.tolist() == [[0.0, 0.0], [0.0, 0.0]]

    arr = np_types.array(_DummyNumpyBackend, [1, 2], dtype=None)
    assert arr.tolist() == [1, 2]

    as_arr = np_types.asarray(_DummyNumpyBackend, [3.5, 4.5])
    assert as_arr.tolist() == [3.5, 4.5]

    val = np_types.item(_DummyNumpyBackend, as_arr[0])
    assert val == 3.5

    # Test numpy_mixins visitors
    visitor = np_mixins.NumpyScatterVisitor()
    node = LogicalNode(id="n1", op_type="ScatterAdd", inputs=["c", "i", "u"])
    vars_list = ["var_c", "var_i", "var_u"]

    res_upd = visitor.visit_TensorScatterUpdate(node, vars_list)
    assert "lambda c, i, u:" in res_upd
    assert "__setitem__" in res_upd

    res_add = visitor.visit_TensorScatterAdd(node, vars_list)
    assert "np.add.at" in res_add

    res_max = visitor.visit_TensorScatterMax(node, vars_list)
    assert "np.maximum.at" in res_max

    res_min = visitor.visit_TensorScatterMin(node, vars_list)
    assert "np.minimum.at" in res_min
