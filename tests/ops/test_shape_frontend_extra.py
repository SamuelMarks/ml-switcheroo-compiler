"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.frontend import dynamic_slice, update_slice


def test_dynamic_slice_indices() -> None:
    """Docstring."""
    config.eager_mode = True
    x = Tensor(np.arange(10), shape=(10,), dtype=DType.Int32, device=Device("cpu"))

    # Tensor indices
    idx_t = Tensor(np.array(2), shape=(), dtype=DType.Int32, device=Device("cpu"))
    res_t = dynamic_slice(x, [idx_t], [3])
    assert np.array_equal(res_t.data, np.array([2, 3, 4]))

    # Int indices
    res_i = dynamic_slice(x, [2], [3])
    assert np.array_equal(res_i.data, np.array([2, 3, 4]))


def test_update_slice_indices() -> None:
    """Docstring."""
    config.eager_mode = True
    x = Tensor(np.arange(5), shape=(5,), dtype=DType.Int32, device=Device("cpu"))
    update = Tensor(np.array([9, 9]), shape=(2,), dtype=DType.Int32, device=Device("cpu"))

    # Tensor indices
    idx_t = Tensor(np.array(1), shape=(), dtype=DType.Int32, device=Device("cpu"))
    res_t = update_slice(x, update, [idx_t])
    assert np.array_equal(res_t.data, np.array([0, 9, 9, 3, 4]))

    # Int indices
    res_i = update_slice(x, update, [1])
    assert np.array_equal(res_i.data, np.array([0, 9, 9, 3, 4]))


def test_shape_frontend_tracing(monkeypatch: object) -> None:
    """Docstring."""
    from unittest.mock import MagicMock

    import ml_switcheroo_compiler.ops.shape.frontend as sf

    config.eager_mode = False

    class MockTracer:
        """Docstring."""

        is_tracing = True
        graph = MagicMock()

        def add_node(self, *args: object, **kwargs: object) -> str:
            """Docstring."""
            return "n1"

    import ml_switcheroo_compiler.ops.shape.utils as su

    monkeypatch.setattr(su, "_tracer", MockTracer())

    class MockData:
        """Docstring."""

        id = "test_id"

    x = Tensor(MockData(), shape=(10,), dtype=DType.Int32, device=Device("cpu"))
    update = Tensor(MockData(), shape=(2,), dtype=DType.Int32, device=Device("cpu"))

    # Test dynamic_slice
    res1 = sf.dynamic_slice(x, [2], [3])
    assert res1 is not None

    # Test update_slice
    res2 = sf.update_slice(x, update, [2])
    assert res2 is not None

    # Test strided_slice
    res3 = sf.strided_slice(x, [0], [3], [1])
    assert res3 is not None

    config.eager_mode = True
