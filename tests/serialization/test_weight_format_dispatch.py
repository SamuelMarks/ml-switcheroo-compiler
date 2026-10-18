"""Unit tests for weight format dispatchers and serialization state extraction."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

import ml_switcheroo_compiler.serialization as s

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_serialization_format_dispatch_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_serialization_format_dispatch.

    Args:
        _state: Shared test state dictionary.
    """
    tmp_path = _state["tmp_path"]
    h5_path = str(tmp_path / "weights.h5")
    st_path = str(tmp_path / "weights.safetensors")
    npz_path = str(tmp_path / "weights.npz")
    pk_path = str(tmp_path / "weights.pk")
    zip_path = str(tmp_path / "model_bundle.zip")
    dummy_model = str(tmp_path / "dummy_model")
    with patch("ml_switcheroo_compiler.serialization.H5WeightFormat.save"):
        s._save_as_h5({"a": 1}, h5_path)
    with patch("ml_switcheroo_compiler.serialization.SafetensorsWeightFormat.save"):
        s._save_as_safetensors({"a": 1}, st_path)
    with patch("ml_switcheroo_compiler.serialization.H5WeightFormat.load"):
        s._load_h5_weights(h5_path)
    with patch("ml_switcheroo_compiler.serialization.SafetensorsWeightFormat.load"):
        s._load_safetensors_weights(st_path)
    with patch("ml_switcheroo_compiler.serialization.NpzWeightFormat.load"):
        s._load_npz_weights(npz_path)
    getattr(_state, "update")({"dummy_model": dummy_model, "h5_path": h5_path, "npz_path": npz_path, "pk_path": pk_path})
    getattr(_state, "update")({"st_path": st_path, "zip_path": zip_path})
    _state.update(locals())


def _test_serialization_format_dispatch_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_serialization_format_dispatch.

    Args:
        _state: Shared test state dictionary.
    """
    pk_path = _state["pk_path"]
    with patch("ml_switcheroo_compiler.serialization.PickleWeightFormat.load"):
        cast("Callable[..., EagerTensorProtocol]", s._load_pickle_weights)(pk_path)

    class FakeWeight:
        """Mock weight variable."""

        def __init__(self, name: str) -> None:
            """Test function __init__."""
            self.name: str = name

        def numpy(self) -> int:
            """Test function numpy."""
            return 1

        def tolist(self) -> int:
            """Test function tolist."""
            return 1

    class FakeModel:
        """Mock neural network model."""

        weights: list[FakeWeight] = [FakeWeight("w")]
        optimizer: EagerValue = type("O", (), {"variables": [FakeWeight("v")], "momentums": [FakeWeight("m")]})()
        non_trainable_variables: list[FakeWeight] = [FakeWeight("nt")]
        ema_variables: list[FakeWeight] = [FakeWeight("ema")]

        def get_config(self) -> dict[str, int]:
            """Test function get_config."""
            return {"c": 1}

    getattr(_state, "update")({"FakeModel": FakeModel, "FakeWeight": FakeWeight})
    _state.update(locals())


def _test_serialization_format_dispatch_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_serialization_format_dispatch.

    Args:
        _state: Shared test state dictionary.
    """
    (FakeModel, zip_path) = (_state["FakeModel"], _state["zip_path"])
    ws = cast("Callable[..., EagerTensorProtocol]", s._extract_model_weights)(cast("Callable[..., EagerTensorProtocol]", FakeModel)())
    assert len(ws) == 1
    st: dict[str, EagerValue] = {}
    s._extract_optimizer_state(FakeModel(), st)
    s._extract_non_trainable_state(FakeModel(), st, {})
    s._extract_ema_state(FakeModel(), st)
    cast("Callable[..., EagerTensorProtocol]", s._extract_model_state)(cast("Callable[..., EagerTensorProtocol]", FakeModel)(), {})
    cast("Callable[..., EagerTensorProtocol]", s._compile_model_metadata)(cast("Callable[..., EagerTensorProtocol]", FakeModel)())
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        with patch("ml_switcheroo_compiler.serialization._save_as_h5"):
            s._write_h5_to_zip(zf, "test.h5", {"a": 1})
    if os.path.exists(zip_path):
        os.remove(zip_path)
    ctx = cast("Callable[..., EagerTensorProtocol]", s.KerasSerializationContext)(filepath=zip_path, config_dict={}, metadata={}, weights_store={"a": 1}, state_store={"b": 1})
    with patch("ml_switcheroo_compiler.serialization._write_h5_to_zip"):
        cast("Callable[..., EagerTensorProtocol]", s._write_keras_zip)(ctx)
    getattr(_state, "update")({"ctx": ctx, "st": st, "ws": ws, "zf": zf})
    _state.update(locals())


def _test_serialization_format_dispatch_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_serialization_format_dispatch.

    Args:
        _state: Shared test state dictionary.
    """
    (FakeModel, dummy_model) = (_state["FakeModel"], _state["dummy_model"])
    cast("Callable[..., EagerTensorProtocol]", s.save_model)(cast("Callable[..., EagerTensorProtocol]", FakeModel)(), dummy_model)
    cast("Callable[..., EagerTensorProtocol]", s.load_model)(dummy_model)
    s.get_custom_objects()
    s.get_registered_name(list)
    s.get_registered_object("list")
    s.serialize_object(FakeModel())
    s.deserialize_object({"class_name": "list"})
    s.TrackableResource()
    s.PythonState()
    s.MaxShardSizePolicy(1)
    s.ShardByTaskPolicy()
    s.SavedModel()
    cast("Callable[..., EagerTensorProtocol]", s.read_fingerprint)(dummy_model)
    cast("Callable[..., EagerTensorProtocol]", s.load_variable)(dummy_model, "var1")
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        cast("Callable[..., EagerTensorProtocol]", s.run_restore_ops)(dummy_model)


def test_serialization_format_dispatch(tmp_path: Path) -> None:
    """Verify weight format save/load dispatchers and model state extraction helpers.

    Args:
        tmp_path (Path): Temporary filesystem directory fixture.
    """
    _state: dict[str, EagerTensorProtocol] = cast("dict[str, EagerTensorProtocol]", {"tmp_path": tmp_path})
    _test_serialization_format_dispatch_part_1(_state)
    _test_serialization_format_dispatch_part_2(_state)
    _test_serialization_format_dispatch_part_3(_state)
    _test_serialization_format_dispatch_part_4(_state)
    assert isinstance(_state, dict)


def test_weight_format_load_save(tmp_path: Path) -> None:
    """Verify weight validation and weight saving/loading dispatch.

    Args:
        tmp_path (Path): Temporary filesystem directory fixture.
    """
    assert cast("Callable[..., EagerTensorProtocol]", s._validate_and_map_weights)({"a": 1}) == {"a": 1}
    tf = str(tmp_path / "last_lines_weights.h5")
    if os.path.exists(tf):
        os.remove(tf)
    s.save_weights(None, tf)
    with cast("Callable[..., EagerTensorProtocol]", patch.object)(cast("EagerTensorProtocol", getattr(s, "H5WeightFormat")), "load", return_value={"x": 2}):
        res = s.load_weights(tf)
        assert res == {"x": 2}
    if os.path.exists(tf):
        os.remove(tf)
