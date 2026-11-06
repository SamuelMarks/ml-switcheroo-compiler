"""Unit tests ensuring 100% test and branch coverage for backend type helper modules."""

from __future__ import annotations

import importlib
import sys
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_dask_types_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 100% line and branch coverage for dask/types.py.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.backends.dask.types as dask_types

    # 1. ImportError branch coverage
    monkeypatch.setitem(sys.modules, "dask", None)
    monkeypatch.setitem(sys.modules, "dask.array", None)
    importlib.reload(dask_types)
    assert ensure_eager_tensor(ensure_eager_tensor(dask_types).da) is None

    # 2. Restore and normal execution
    monkeypatch.undo()
    importlib.reload(dask_types)
    if ensure_eager_tensor(ensure_eager_tensor(dask_types).da) is None:
        import types

        mock_dask = types.ModuleType("dask")
        mock_da = types.ModuleType("dask.array")

        class _MockDaskArr:
            """Mock Dask array."""

            def __init__(self, arr: EagerValue) -> None:
                """Initialize _MockDaskArr.

                Args:
                    arr: Underlying numpy array.
                """
                self._arr = np.asarray(arr)

            def compute(self) -> EagerValue:
                """Compute numpy array.

                Returns:
                    EagerValue: Computed array.
                """
                return ensure_eager_tensor(self._arr)

        ensure_mutable(mock_da).zeros = ensure_eager_tensor(lambda shape, **kw: _MockDaskArr(np.zeros(shape)))
        ensure_mutable(mock_da).array = ensure_eager_tensor(lambda data, dtype=None, **kw: _MockDaskArr(np.array(data, dtype=dtype)))
        ensure_mutable(mock_da).asarray = ensure_eager_tensor(lambda data, **kw: _MockDaskArr(np.asarray(data)))
        ensure_mutable(mock_dask).array = ensure_eager_tensor(mock_da)
        monkeypatch.setitem(sys.modules, "dask", mock_dask)
        monkeypatch.setitem(sys.modules, "dask.array", mock_da)
        importlib.reload(dask_types)
    assert ensure_eager_tensor(ensure_eager_tensor(dask_types).da) is not None

    z = dask_types.zeros(type, (2, 2))
    assert z is not None
    arr_no_dtype = dask_types.array(type, [1, 2, 3])
    assert arr_no_dtype is not None
    arr_dtype = dask_types.array(type, [1.0, 2.0], dtype="float32")
    assert arr_dtype is not None
    as_arr = dask_types.asarray(type, [4, 5])
    assert as_arr is not None
    val = dask_types.item(type, np.array(42.0))
    assert val == 42.0
    monkeypatch.undo()
    importlib.reload(dask_types)


def test_mlx_types_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 100% line and branch coverage for mlx/types.py.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.backends.mlx.types as mlx_types

    # 1. ImportError branch coverage
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    importlib.reload(mlx_types)
    assert ensure_eager_tensor(mlx_types.mx) is None

    # 2. Restore and normal execution
    monkeypatch.undo()
    importlib.reload(mlx_types)
    if ensure_eager_tensor(mlx_types.mx) is None:
        import types

        mock_mlx = types.ModuleType("mlx")
        mock_mx = types.ModuleType("mlx.core")
        ensure_mutable(mock_mx).zeros = ensure_eager_tensor(np.zeros)
        ensure_mutable(mock_mx).array = ensure_eager_tensor(np.array)
        ensure_mutable(mock_mx).float32 = ensure_eager_tensor(np.float32)
        ensure_mutable(mock_mlx).core = ensure_eager_tensor(mock_mx)
        monkeypatch.setitem(sys.modules, "mlx", mock_mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", mock_mx)
        importlib.reload(mlx_types)
    assert ensure_eager_tensor(mlx_types.mx) is not None

    z = mlx_types.zeros(type, (2, 2))
    assert z is not None
    arr = mlx_types.array(type, [1, 2, 3], dtype=None)
    assert arr is not None
    arr_str_dtype = mlx_types.array(type, [1.0, 2.0], dtype="float32")
    assert arr_str_dtype is not None
    as_arr = mlx_types.asarray(type, [4, 5])
    assert as_arr is not None
    val = mlx_types.item(type, np.array(10.0))
    assert val == 10.0

    # 3. Branch when mx has asarray
    class MockMxWithAsarray:
        """Mock mlx module with asarray function."""

        @staticmethod
        def asarray(data: EagerValue) -> EagerValue:
            """Create asarray.

            Args:
                data: Test parameter `data`.
            """
            return np.array(data)

    monkeypatch.setattr(mlx_types, "mx", MockMxWithAsarray)
    res_mx_asarray = mlx_types.asarray(type, [1, 2])
    assert res_mx_asarray is not None

    # 4. Fallback branch when mx does not have asarray (native mlx)
    monkeypatch.setattr(mlx_types, "mx", mlx_types.mx)
    res_native = mlx_types.asarray(type, [1, 2])
    assert res_native is not None
    monkeypatch.undo()
    importlib.reload(mlx_types)


def test_keras_types_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 100% line and branch coverage for keras/types.py.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.backends.keras.types as keras_types

    # 1. Exception/ImportError branch coverage
    monkeypatch.setitem(sys.modules, "keras", None)
    monkeypatch.setitem(sys.modules, "keras.ops", None)
    importlib.reload(keras_types)
    assert ensure_eager_tensor(ensure_eager_tensor(keras_types).kops) is None

    # 2. Restore and normal execution
    monkeypatch.undo()
    importlib.reload(keras_types)
    if ensure_eager_tensor(ensure_eager_tensor(keras_types).kops) is None:
        import types

        mock_keras = types.ModuleType("keras")
        mock_kops = types.ModuleType("keras.ops")
        ensure_mutable(mock_kops).zeros = ensure_eager_tensor(np.zeros)
        ensure_mutable(mock_kops).array = ensure_eager_tensor(np.array)
        ensure_mutable(mock_kops).convert_to_tensor = ensure_eager_tensor(np.asarray)
        ensure_mutable(mock_keras).ops = ensure_eager_tensor(mock_kops)
        monkeypatch.setitem(sys.modules, "keras", mock_keras)
        monkeypatch.setitem(sys.modules, "keras.ops", mock_kops)
        importlib.reload(keras_types)
    assert ensure_eager_tensor(ensure_eager_tensor(keras_types).kops) is not None

    z = keras_types.zeros(type, (2, 2))
    assert z is not None
    arr = keras_types.array(type, [1, 2, 3], dtype="float32")
    assert arr is not None
    as_arr = keras_types.asarray(type, [4, 5])
    assert as_arr is not None
    val = keras_types.item(type, np.array(7.0))
    assert val == 7.0

    # 3. Test asarray when kops has asarray
    class MockKopsWithAsarray:
        """Mock kops with asarray method."""

        @staticmethod
        def asarray(data: EagerValue) -> EagerValue:
            """Asarray.

            Args:
                data: Test parameter `data`.
            """
            return np.array(data)

    monkeypatch.setattr(keras_types, "kops", MockKopsWithAsarray)
    res_kops_asarray = keras_types.asarray(type, [1, 2])
    assert res_kops_asarray is not None
    monkeypatch.undo()
    importlib.reload(keras_types)
