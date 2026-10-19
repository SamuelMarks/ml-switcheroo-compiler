# ruff: noqa: E501
"""Tests for H5 serialization format and tensor conversion branches."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_h5_branches() -> None:
    """Test function test_h5_branches."""
    import os

    import numpy as np

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    ser = H5WeightFormat()
    test_file = "test_h5_serialization_branches.h5"
    if os.path.exists(test_file):
        os.remove(test_file)

    class FakeTensor1:
        """Test class FakeTensor1."""

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return np.array([1, 2])

    class FakeTensor2:
        """Test class FakeTensor2."""

        def __init__(self) -> None:
            """Test function __init__."""

            class D:
                """Test class D."""

                def numpy(self) -> EagerValue:
                    """Test function numpy."""
                    return np.array([3, 4])

            self.data = D()

    class FakeTensor3:
        """Test class FakeTensor3."""

        def tolist(self) -> EagerValue:
            """Test function tolist."""
            return [5, 6]

    weights = {"t1": FakeTensor1(), "t2": FakeTensor2(), "t3": FakeTensor3()}
    import ml_switcheroo_compiler.backends.registry as reg

    old = reg.get_active_backend

    class FakeBackend:
        """Test class FakeBackend."""

        pass

    setattr(reg, "get_active_backend", lambda: FakeBackend())
    cast("EagerCallableProtocol", ser.save)(weights, test_file)
    loaded = ser.load(test_file)
    assert cast("EagerCallableProtocol", np.array_equal)(loaded["t1"], [1, 2])
    assert cast("EagerCallableProtocol", np.array_equal)(loaded["t2"], [3, 4])
    assert cast("EagerCallableProtocol", np.array_equal)(loaded["t3"], [5, 6])
    reg.get_active_backend = old
    if os.path.exists(test_file):
        os.remove(test_file)


def test_h5_backend_branches() -> None:
    """Test function test_h5_backend_branches."""
    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    ser = H5WeightFormat()

    class FakeBackend:
        """Test class FakeBackend."""

        def load_h5(self, path: EagerValue) -> EagerValue:
            """Test function load_h5.

            Args:
                path: Test parameter `path`.
            """
            return {"a": 1}

        def save_h5(self, weights: EagerValue, path: EagerValue) -> None:
            """Test function save_h5.

            Args:
                weights: Test parameter `weights`.
                path: Test parameter `path`.
            """
            pass

    import ml_switcheroo_compiler.backends.registry as reg

    old = reg.get_active_backend
    setattr(reg, "get_active_backend", lambda: FakeBackend())
    assert cast("EagerTensorProtocol", ser.load("dummy.h5")) == cast("EagerTensorProtocol", {"a": 1})
    cast("EagerCallableProtocol", ser.save)({"a": 1}, "dummy.h5")
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
    cast("EagerCallableProtocol", ser.save)({"plain": cast("EagerCallableProtocol", np.array)(5)}, test_file)
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

    cast("EagerCallableProtocol", ser.save)({"plain2": 42}, test_file)
    loaded = ser.load(test_file)
    assert cast("EagerTensorProtocol", loaded["plain2"]) == cast("EagerTensorProtocol", 42)
    if os.path.exists(test_file):
        os.remove(test_file)


def test_h5_missing_h5py(monkeypatch: EagerTensorProtocol) -> None:
    """Test that H5WeightFormat raises ImportError when h5py is None.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import numpy as np
    import pytest

    import ml_switcheroo_compiler.serialization.formats.h5 as h5_mod

    monkeypatch.setattr(h5_mod, "h5py", None)
    ser = h5_mod.H5WeightFormat()
    with pytest.raises(ImportError, match="h5py is required for H5 weight loading"):
        ser.load("dummy.h5")

    with pytest.raises(ImportError, match="h5py is required for H5 weight saving"):
        cast("EagerCallableProtocol", ser.save)({"a": cast("EagerCallableProtocol", np.array)([1])}, "dummy.h5")
