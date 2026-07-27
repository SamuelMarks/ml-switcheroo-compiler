"""Tests for serialization stubs."""

from ml_switcheroo_compiler.serialization import (
    deserialize_keras_object,
    get_registered_object,
    graph_to_json,
    load_model,
    run_restore_ops,
    serialize_keras_object,
)


def test_serialization_stubs() -> None:
    """Test serialization stubs."""
    try:
        run_restore_ops("path")
    except FileNotFoundError:
        pass
    assert get_registered_object() is None
    assert deserialize_keras_object() == {}
    assert serialize_keras_object() == {}

    # load_model returns FallbackModel now
    model = load_model("path")
    assert model.__class__.__name__ == "FallbackModel"


def test_file_handler() -> None:
    """Test file handler."""

    class DummyGraph:
        def to_json(self):
            return "{}"

    assert graph_to_json(DummyGraph()) == "{}"


def test_msgpack() -> None:
    """Test msgpack handler."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.serialization import MsgpackWeightFormat

    fmt = MsgpackWeightFormat()
    import pytest

    with patch.dict("sys.modules", {"msgpack": None}):
        with pytest.raises(ImportError):
            fmt.load("path")
        with pytest.raises(ImportError):
            fmt.save({}, "path")
