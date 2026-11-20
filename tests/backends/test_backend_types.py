"""Unit tests ensuring 100% test and branch coverage for backend type helper modules."""

from __future__ import annotations

import importlib
import sys
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


def test_dask_types_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 100% line and branch coverage for dask/types.py.

    Args:
        monkeypatch (pytest.MonkeyPatch): Test parameter `monkeypatch`.
    """
    import ml_switcheroo_compiler.backends.dask.types as dask_types

    # 1. ImportError branch coverage
    monkeypatch.setitem(sys.modules, "dask", None)
    monkeypatch.setitem(sys.modules, "dask.array", None)
    importlib.reload(dask_types)
    assert adapt_backend_module(dask_types).resolve_symbol("da")[1] is None

    # 2. Restore and normal execution
    monkeypatch.undo()
    importlib.reload(dask_types)
    if adapt_backend_module(dask_types).resolve_symbol("da")[1] is None:
        import types

        mock_dask = types.ModuleType("dask")
        mock_da = types.ModuleType("dask.array")

        class _MockDaskArr:
            """Mock Dask array."""

            def __init__(self, arr: BackendArg) -> None:
                """Initialize _MockDaskArr.

                Args:
                    arr (BackendArg): Underlying numpy array.
                """
                self._arr = np.asarray(arr)

            def compute(self) -> BackendArg:
                """Compute numpy array.

                Returns:
                    BackendArg: Computed array.
                """
                return self._arr

        pytest.MonkeyPatch().setattr(mock_da, "zeros", lambda shape, **kw: _MockDaskArr(np.zeros(shape)), raising=False)
        pytest.MonkeyPatch().setattr(
            mock_da, "array", lambda data, dtype=None, **kw: _MockDaskArr(np.array(data, dtype=dtype)), raising=False
        )
        pytest.MonkeyPatch().setattr(
            mock_da, "asarray", lambda data, **kw: _MockDaskArr(np.asarray(data)), raising=False
        )
        pytest.MonkeyPatch().setattr(mock_dask, "array", mock_da, raising=False)
        monkeypatch.setitem(sys.modules, "dask", mock_dask)
        monkeypatch.setitem(sys.modules, "dask.array", mock_da)
        importlib.reload(dask_types)
    assert adapt_backend_module(dask_types).resolve_symbol("da")[1] is not None

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
        monkeypatch (pytest.MonkeyPatch): Test parameter `monkeypatch`.
    """
    import types

    import ml_switcheroo_compiler.backends.mlx.types as mlx_types

    @typing.runtime_checkable
    class _MlxModProto(typing.Protocol):
        """Protocol exposing `mx` module attribute."""

        @property
        def mx(self) -> types.ModuleType | None:
            """Return `mx` module reference.

            Returns:
                types.ModuleType | None: Current `mx` attribute.
            """
            ...

    def _get_mx(m: types.ModuleType) -> types.ModuleType | None:
        """Read `m.mx` without narrowing across reload.

        Args:
            m (types.ModuleType): MLX types module.

        Returns:
            types.ModuleType | None: Current `mx` module reference.
        """
        assert isinstance(m, _MlxModProto)
        val = m.mx
        return val if isinstance(val, types.ModuleType) else None

    # 1. Exception/ImportError branch coverage
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    importlib.reload(mlx_types)
    assert _get_mx(mlx_types) is None

    # 2. Restore and normal execution
    monkeypatch.undo()
    importlib.reload(mlx_types)
    if _get_mx(mlx_types) is None:
        import types

        mock_mlx = types.ModuleType("mlx")
        mock_mx = types.ModuleType("mlx.core")
        pytest.MonkeyPatch().setattr(mock_mx, "zeros", np.zeros, raising=False)
        pytest.MonkeyPatch().setattr(mock_mx, "array", np.array, raising=False)
        pytest.MonkeyPatch().setattr(mock_mx, "float32", np.float32, raising=False)
        pytest.MonkeyPatch().setattr(mock_mlx, "core", mock_mx, raising=False)
        monkeypatch.setitem(sys.modules, "mlx", mock_mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", mock_mx)
        importlib.reload(mlx_types)
    orig_mx = _get_mx(mlx_types)
    assert orig_mx is not None

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
        def asarray(data: BackendArg) -> BackendArg:
            """Create asarray.

            Args:
                data (BackendArg): Test parameter `data`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return np.array(data)

    monkeypatch.setattr(mlx_types, "mx", MockMxWithAsarray, raising=False)
    res_mx_asarray = mlx_types.asarray(type, [1, 2])
    assert res_mx_asarray is not None

    # 4. Fallback branch when mx does not have asarray (native mlx)
    monkeypatch.setattr(mlx_types, "mx", orig_mx, raising=False)
    res_native = mlx_types.asarray(type, [1, 2])
    assert res_native is not None
    monkeypatch.undo()
    importlib.reload(mlx_types)


def test_keras_types_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 100% line and branch coverage for keras/types.py.

    Args:
        monkeypatch (pytest.MonkeyPatch): Test parameter `monkeypatch`.
    """
    import types

    import ml_switcheroo_compiler.backends.keras.types as keras_types

    @typing.runtime_checkable
    class _KopsModProto(typing.Protocol):
        """Protocol exposing `kops` module attribute."""

        @property
        def kops(self) -> types.ModuleType | None:
            """Return `kops` module reference.

            Returns:
                types.ModuleType | None: Current `kops` attribute.
            """
            ...

    def _get_kops(m: types.ModuleType) -> types.ModuleType | None:
        """Read `kops` attribute via `_KopsModProto`.

        Args:
            m (types.ModuleType): Keras types module.

        Returns:
            types.ModuleType | None: Current `kops` attribute.
        """
        assert isinstance(m, _KopsModProto)
        val = m.kops
        return val if isinstance(val, types.ModuleType) else None

    # 1. Exception/ImportError branch coverage
    monkeypatch.setitem(sys.modules, "keras", None)
    monkeypatch.setitem(sys.modules, "keras.ops", None)
    importlib.reload(keras_types)
    assert _get_kops(keras_types) is None

    # 2. Restore and normal execution
    monkeypatch.undo()
    importlib.reload(keras_types)
    if _get_kops(keras_types) is None:
        import types

        mock_keras = types.ModuleType("keras")
        mock_kops = types.ModuleType("keras.ops")
        pytest.MonkeyPatch().setattr(mock_kops, "zeros", np.zeros, raising=False)
        pytest.MonkeyPatch().setattr(mock_kops, "array", np.array, raising=False)
        pytest.MonkeyPatch().setattr(mock_kops, "convert_to_tensor", np.asarray, raising=False)
        pytest.MonkeyPatch().setattr(mock_keras, "ops", mock_kops, raising=False)
        monkeypatch.setitem(sys.modules, "keras", mock_keras)
        monkeypatch.setitem(sys.modules, "keras.ops", mock_kops)
        importlib.reload(keras_types)
    assert _get_kops(keras_types) is not None

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
        def asarray(data: BackendArg) -> BackendArg:
            """Asarray.

            Args:
                data (BackendArg): Test parameter `data`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return np.array(data)

    monkeypatch.setattr(keras_types, "kops", MockKopsWithAsarray, raising=False)
    res_kops_asarray = keras_types.asarray(type, [1, 2])
    assert res_kops_asarray is not None
    monkeypatch.undo()
    importlib.reload(keras_types)
