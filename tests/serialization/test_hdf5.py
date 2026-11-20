"""Test HDF5 serialization format."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import typing
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest

import ml_switcheroo_compiler.serialization.formats.h5 as hdf5_mod
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.serialization.formats.h5 import HDF5WeightLoader, HDF5WeightSaver

if typing.TYPE_CHECKING:
    pass


def test_hdf5_save_and_load() -> None:
    """Test saving and loading weights using HDF5 format."""
    weights = {
        "layer1/weight": np.random.rand(10, 10),
        "layer1/bias": np.random.rand(10),
        "layer2/weight": np.random.rand(5, 10),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.h5")

        saver = HDF5WeightSaver()
        saver.save(weights, filepath)

        assert os.path.exists(filepath)

        loader = HDF5WeightLoader()
        loaded_weights = loader.load(filepath)

        assert len(loaded_weights) == 3
        for k, v in weights.items():
            assert k in loaded_weights
            np.testing.assert_allclose(ensure_ndarray(loaded_weights[k]), v)


def test_hdf5_schema_validation() -> None:
    """Test schema validation for HDF5."""
    from ml_switcheroo_compiler.serialization.formats.h5 import WeightSchema

    # Should construct successfully
    schema = WeightSchema(data={"a": np.array([1, 2, 3])})
    assert "a" in schema.data


def test_hdf5_missing_h5py(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that HDF5WeightLoader and HDF5WeightSaver raise ImportError when h5py is None.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import pytest

    import ml_switcheroo_compiler.serialization.formats.h5 as hdf5_mod

    monkeypatch.setattr(hdf5_mod, "h5py", None)
    loader = hdf5_mod.HDF5WeightLoader()
    with pytest.raises(ImportError, match="h5py is required for HDF5 weight loading"):
        loader.load("dummy.h5")

    saver = hdf5_mod.HDF5WeightSaver()
    with pytest.raises(ImportError, match="h5py is required for HDF5 weight saving"):
        saver.save({"a": np.array([1])}, "dummy.h5")


def test_h5_and_hdf5_import_error_coverage(tmp_path: Path) -> None:
    """Test fallback import branch when h5py is missing.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    import importlib
    import sys
    from unittest import mock

    fake_h5_file = tmp_path / "test.h5"
    fake_h5_file.write_bytes(b"placeholder")

    with mock.patch.dict(sys.modules, {"h5py": None}):
        import ml_switcheroo_compiler.serialization.formats.h5 as h5_mod
        import ml_switcheroo_compiler.serialization.formats.h5 as hdf5_mod

        importlib.reload(h5_mod)
        importlib.reload(hdf5_mod)

        assert h5_mod.h5py is None
        assert hdf5_mod.h5py is None

    import ml_switcheroo_compiler.serialization.formats.h5 as h5_mod
    import ml_switcheroo_compiler.serialization.formats.h5 as hdf5_mod

    importlib.reload(h5_mod)
    importlib.reload(hdf5_mod)


class _DummyGenerator:
    """Mock generator providing fallback prefix."""

    def get_fallback_prefix(self) -> str:
        """Return prefix for fallback ops.

        Returns:
            str: Prefix string.
        """
        return "np_eager"


class _MockArrayWithNumpy:
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


class _MockDataWithNumpy:
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


class _MockArrayWithToList:
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


def test_hdf5_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HDF5WeightLoader and HDF5WeightSaver with all branches.

    Args:
        tmp_path (Path): Pytest temporary path.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.

    Returns:
        None
    """
    # 1. Test missing h5py module import branch via reload
    monkeypatch.setitem(sys.modules, "h5py", None)
    importlib.reload(hdf5_mod)
    assert hdf5_mod.h5py is None

    # Test error raising when h5py is None
    loader_missing = HDF5WeightLoader()
    saver_missing = HDF5WeightSaver()
    dummy_path = str(tmp_path / "dummy.h5")
    with pytest.raises(ImportError, match="h5py is required"):
        loader_missing.load(dummy_path)
    with pytest.raises(ImportError, match="h5py is required"):
        saver_missing.save({}, dummy_path)

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
            m (types.ModuleType): Target hdf5 format module.

        Returns:
            types.ModuleType | None: Current `h5py` module attribute.
        """
        assert isinstance(m, _H5ModProto)
        val = m.h5py
        return val if isinstance(val, types.ModuleType) else None

    # Restore h5py
    monkeypatch.undo()
    importlib.reload(hdf5_mod)
    assert _get_h5py(hdf5_mod) is not None

    loader = HDF5WeightLoader()
    saver = HDF5WeightSaver()
    filepath = str(tmp_path / "weights.h5")

    data = {"w": np.array([1.0, 2.0], dtype=np.float32)}
    saver.save(data, filepath)

    # Insert a subgroup and dataset to exercise both branches of isinstance(node, h5py.Dataset)
    with h5py.File(filepath, "a") as f:
        grp = f.create_group("subgroup")
        grp.create_dataset("b", data=np.array([3.0, 4.0], dtype=np.float32))

    loaded = loader.load(filepath)
    assert "w" in loaded
    assert "subgroup/b" in loaded
    np.testing.assert_allclose(np.asarray(loaded["w"]), data["w"])
