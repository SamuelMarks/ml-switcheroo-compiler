"""Behavioral unit tests for h5 serialization."""

from __future__ import annotations

import importlib
import sys
import typing
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest

import ml_switcheroo_compiler.backends.registry as registry
import ml_switcheroo_compiler.serialization.formats.h5 as h5_mod
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat
from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_h5_branches() -> None:
    """Test function test_h5_branches."""
    import os

    import numpy as np

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    ser = H5WeightFormat()
    test_file = "test_h5_serialization_branches.h5"
    if os.path.exists(test_file):
        os.remove(test_file)

    from tests.testing_utils import MockTestStub

    class FakeTensor1(MockTestStub):
        """Test class FakeTensor1."""

        def numpy(self) -> BackendArg:
            """Test function numpy.

            Returns:
                BackendArg: Computed output value from `numpy`.
            """
            return np.array([1, 2])

    class FakeTensor2(MockTestStub):
        """Test class FakeTensor2."""

        def __init__(self) -> None:
            """Test function __init__."""

            class D:
                """Test class D."""

                def numpy(self) -> BackendArg:
                    """Test function numpy.

                    Returns:
                        BackendArg: Computed output value from `numpy`.
                    """
                    return np.array([3, 4])

            self.data = D()

    class FakeTensor3(MockTestStub):
        """Test class FakeTensor3."""

        def tolist(self) -> list[int]:
            """Test function tolist.

            Returns:
                list[int]: Computed output value from `tolist`.
            """
            return [5, 6]

    from ml_switcheroo_compiler.core.shape import OpValue

    weights: dict[str, OpValue] = {"t1": FakeTensor1(), "t2": FakeTensor2(), "t3": FakeTensor3()}
    import ml_switcheroo_compiler.backends.registry as reg

    old = reg.get_active_backend

    class FakeBackend:
        """Test class FakeBackend."""

        pass

    pytest.MonkeyPatch().setattr(reg, "get_active_backend", lambda: FakeBackend(), raising=False)
    ser.save(weights, test_file)
    loaded = ser.load(test_file)
    assert np.array_equal(ensure_ndarray(loaded["t1"]), [1, 2])
    assert np.array_equal(ensure_ndarray(loaded["t2"]), [3, 4])
    assert np.array_equal(ensure_ndarray(loaded["t3"]), [5, 6])
    reg.get_active_backend = old
    if os.path.exists(test_file):
        os.remove(test_file)


def test_h5_backend_branches() -> None:
    """Test function test_h5_backend_branches."""
    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    ser = H5WeightFormat()

    class FakeBackend:
        """Test class FakeBackend."""

        def load_h5(self, path: BackendArg) -> BackendArg:
            """Test function load_h5.

            Args:
                path: Test parameter `path`.

            Returns:
                BackendArg: Computed output value from `load_h5`.
            """
            return {"a": 1}

        def save_h5(self, weights: BackendArg, path: BackendArg) -> None:
            """Test function save_h5.

            Args:
                weights: Test parameter `weights`.
                path: Test parameter `path`.
            """
            pass

    import ml_switcheroo_compiler.backends.registry as reg

    old = reg.get_active_backend
    pytest.MonkeyPatch().setattr(reg, "get_active_backend", lambda: FakeBackend(), raising=False)
    assert ser.load("dummy.h5") == {"a": 1}
    ser.save({"a": 1}, "dummy.h5")
    reg.get_active_backend = old


def test_h5_visit_not_dataset() -> None:
    """Test function test_h5_visit_not_dataset."""
    import os

    import h5py

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    test_file = "test_h5_group.h5"
    if os.path.exists(test_file):
        os.remove(test_file)
    with h5py.File(test_file, "w") as f:
        f.create_group("my_group")
    ser = H5WeightFormat()
    loaded = ser.load(test_file)
    assert "my_group" not in loaded
    if os.path.exists(test_file):
        os.remove(test_file)


def test_h5_save_plain() -> None:
    """Test function test_h5_save_plain."""
    import os

    import numpy as np

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    test_file = "test_h5_plain.h5"
    if os.path.exists(test_file):
        os.remove(test_file)
    ser = H5WeightFormat()
    ser.save({"plain": np.array(5)}, test_file)
    loaded = ser.load(test_file)
    assert "plain" in loaded
    if os.path.exists(test_file):
        os.remove(test_file)


def test_h5_save_no_branches_taken() -> None:
    """Test function test_h5_save_no_branches_taken."""
    import os

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    test_file = "test_h5_plain2.h5"
    if os.path.exists(test_file):
        os.remove(test_file)
    ser = H5WeightFormat()

    class FakeNoAttribs:
        """Test class FakeNoAttribs."""

        pass

    ser.save({"plain2": 42}, test_file)
    loaded = ser.load(test_file)
    assert loaded["plain2"] == 42
    if os.path.exists(test_file):
        os.remove(test_file)


def test_h5_missing_h5py(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that H5WeightFormat raises ImportError when h5py is None.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import numpy as np
    import pytest

    import ml_switcheroo_compiler.serialization.formats.h5 as h5_mod

    monkeypatch.setattr(h5_mod, "h5py", None)
    ser = h5_mod.H5WeightFormat()
    with pytest.raises(ImportError, match="h5py is required for HDF5 weight loading"):
        ser.load("dummy.h5")

    with pytest.raises(ImportError, match="h5py is required for HDF5 weight saving"):
        ser.save({"a": np.array([1])}, "dummy.h5")


class _DummyGenerator:
    """Mock generator providing fallback prefix."""

    def get_fallback_prefix(self) -> str:
        """Return prefix for fallback ops.

        Returns:
            str: Prefix string.
        """
        return "np_eager"


class _MockArrayWithNumpy(MockTestStub):
    """Mock object with numpy() method."""

    def __init__(
        self,
        arr: np.ndarray[
            tuple[int, ...],
            np.dtype[np.generic],
        ],
    ) -> None:
        """Initialize with numpy array.

        Args:
            arr (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Underlying array.
        """
        self.arr = arr

    def numpy(
        self,
    ) -> np.ndarray[
        tuple[int, ...],
        np.dtype[np.generic],
    ]:
        """Return underlying numpy array.

        Returns:
            np.ndarray: Array.
        """
        return self.arr


class _MockDataWithNumpy(MockTestStub):
    """Mock object with data.numpy() method."""

    def __init__(
        self,
        arr: np.ndarray[
            tuple[int, ...],
            np.dtype[np.generic],
        ],
    ) -> None:
        """Initialize with nested data object.

        Args:
            arr (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Underlying array.
        """
        self.data = _MockArrayWithNumpy(arr)


class _MockArrayWithToList(MockTestStub):
    """Mock object with tolist() method."""

    def __init__(self, lst: list[float]) -> None:
        """Initialize with list.

        Args:
            lst (list[float]): List elements.
        """
        self.lst = lst

    def tolist(self) -> list[float]:
        """Return list representation.

        Returns:
            list[float]: The list.
        """
        return self.lst


class _NoShapeObject:
    """Mock object with dtype and tobytes, but missing shape."""

    dtype: str = "float32"

    def tobytes(self) -> bytes:
        """Return empty bytes.

        Returns:
            bytes: Empty bytes.
        """
        return b""


class _NoTobytesObject:
    """Mock object with dtype and shape, but missing tobytes."""

    dtype: str = "float32"
    shape: tuple[int, ...] = (2,)


class _MockBackendH5:
    """Mock backend implementing custom h5 save and load methods."""

    def __init__(self) -> None:
        """Initialize mock backend."""
        self.saved: (
            tuple[
                dict[
                    str,
                    np.ndarray[
                        tuple[int, ...],
                        np.dtype[np.generic],
                    ],
                ],
                str,
            ]
            | None
        ) = None

    def load_h5(
        self, path: str
    ) -> dict[
        str,
        np.ndarray[
            tuple[int, ...],
            np.dtype[np.generic],
        ],
    ]:
        """Load h5 mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([42.0])}

    def save_h5(
        self,
        weights: dict[
            str,
            np.ndarray[
                tuple[int, ...],
                np.dtype[np.generic],
            ],
        ],
        path: str,
    ) -> None:
        """Save h5 mock weights.

        Args:
            weights (dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)


class _MockBackendNPZ:
    """Mock backend implementing custom npz save and load methods."""

    def __init__(self) -> None:
        """Initialize mock backend."""
        self.saved: (
            tuple[
                dict[
                    str,
                    np.ndarray[
                        tuple[int, ...],
                        np.dtype[np.generic],
                    ],
                ],
                str,
            ]
            | None
        ) = None

    def save_npz(
        self,
        weights: dict[
            str,
            np.ndarray[
                tuple[int, ...],
                np.dtype[np.generic],
            ],
        ],
        path: str,
    ) -> None:
        """Save npz mock weights.

        Args:
            weights (dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)

    def load_npz(
        self, path: str
    ) -> dict[
        str,
        np.ndarray[
            tuple[int, ...],
            np.dtype[np.generic],
        ],
    ]:
        """Load npz mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([100.0])}


class _MockBackendSafetensors:
    """Mock backend implementing custom from_buffer method."""

    def from_buffer(
        self, buffer: bytes, dtype: str, shape: Sequence[int]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        """Reconstruct array from buffer.

        Args:
            buffer (bytes): Raw bytes.
            dtype (str): Dtype identifier.
            shape (Sequence[int]): Shape tuple or list.

        Returns:
            np.ndarray: Reconstructed array.
        """
        return np.frombuffer(buffer, dtype=np.float32).reshape(tuple(shape))


class _PlainBackend:
    """Plain backend without custom format hooks."""

    pass


def test_h5_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify H5WeightFormat with all data conversions and backend hooks.

    Args:
        tmp_path (Path): Pytest temporary path.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.

    Returns:
        None
    """
    import types

    @typing.runtime_checkable
    class _H5ModProto(typing.Protocol):
        """Protocol exposing `h5py` module attribute."""

        @property
        def h5py(self) -> types.ModuleType | None:
            """Return `h5py` module attribute.

            Returns:
                types.ModuleType | None: Current `h5py` module.
            """
            ...

    def _get_h5py(m: types.ModuleType) -> types.ModuleType | None:
        """Read `m.h5py` without narrowing across `importlib.reload`.

        Args:
            m (types.ModuleType): Target h5 format module.

        Returns:
            types.ModuleType | None: Current `h5py` module attribute.
        """
        assert isinstance(m, _H5ModProto)
        val = m.h5py
        return val if isinstance(val, types.ModuleType) else None

    # 1. Missing h5py module import branch via reload
    monkeypatch.setitem(sys.modules, "h5py", None)
    importlib.reload(h5_mod)
    assert _get_h5py(h5_mod) is None

    fmt_missing = H5WeightFormat()
    dummy_path = str(tmp_path / "missing.h5")
    with pytest.raises(ImportError, match="h5py is required"):
        fmt_missing.load(dummy_path)
    with pytest.raises(ImportError, match="h5py is required"):
        fmt_missing.save({}, dummy_path)

    # Restore h5py
    monkeypatch.undo()
    importlib.reload(h5_mod)
    assert _get_h5py(h5_mod) is not None

    fmt = H5WeightFormat()

    # 2. Backend hooks load_h5 and save_h5
    mock_b = _MockBackendH5()
    monkeypatch.setattr(registry, "get_active_backend", lambda: mock_b)
    res = fmt.load(dummy_path)
    assert "backend_key" in res
    fmt.save({"w": np.array([1.0])}, dummy_path)
    assert mock_b.saved is not None
    assert mock_b.saved[1] == dummy_path

    # 3. Plain backend without custom hooks
    monkeypatch.setattr(registry, "get_active_backend", lambda: _PlainBackend())

    from ml_switcheroo_compiler.core.shape import OpValue

    filepath = str(tmp_path / "actual.h5")
    data: dict[str, OpValue] = {
        "v_arr": np.array([1.0, 2.0]),
        "v_numpy": _MockArrayWithNumpy(np.array([3.0, 4.0])),
        "v_data_numpy": _MockDataWithNumpy(np.array([5.0, 6.0])),
        "v_tolist": _MockArrayWithToList([7.0, 8.0]),
        "v_raw": 100,
    }
    fmt.save(data, filepath)

    # Insert a group to exercise both branches of dataset visiting
    with h5py.File(filepath, "a") as f:
        grp = f.create_group("subgroup")
        grp.create_dataset("nested", data=np.array([9.0]))

    loaded = fmt.load(filepath)
    assert "v_arr" in loaded
    assert "v_numpy" in loaded
    assert "v_data_numpy" in loaded
    assert "v_tolist" in loaded
    assert "v_raw" in loaded
    assert "subgroup/nested" in loaded
