"""Tests for test_generated_registry."""

from __future__ import annotations

import typing
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY
from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY, get_op
from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_generated_registry_loaded() -> None:
    """Test function test_generated_registry_loaded."""
    assert isinstance(OPS_REGISTRY, dict)
    assert "Abs" in OPS_REGISTRY


def test_dynamic_infer_shape() -> None:
    """Test function test_dynamic_infer_shape."""
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY

    fake_entry: dict[str, dict[str, dict[str, str]]] = {
        "metadata": {"operation": {"name": "FakeShapeOp"}},
    }
    original = _YAML_REGISTRY.get("FakeShapeOp")
    _YAML_REGISTRY["FakeShapeOp"] = fake_entry

    try:
        OpCls = get_op("FakeShapeOp")
        op = OpCls()

        # 1. inputs via list of args
        class Dummy(MockTestStub):
            """Test class Dummy."""

            pass

        d1 = Dummy()
        pytest.MonkeyPatch().setattr(d1, "shape_metadata", (2, 2), raising=False)
        d2 = Dummy()
        pytest.MonkeyPatch().setattr(d2, "shape", (2, 2), raising=False)

        assert op.dispatch_infer_shape(d1) == (2, 2)
        assert op.dispatch_infer_shape(inputs=[d2]) == (2, 2)

        @typing.runtime_checkable
        class _YamlDataProto(typing.Protocol):
            """Protocol for DynamicOpDef exposing `get_yaml_data`."""

            def get_yaml_data(self) -> BackendArg:
                """Return backing YAML entry.

                Returns:
                    BackendArg: YAML dictionary.
                """
                ...

        # 2. list of ints
        assert op.dispatch_infer_shape([[2, 3]]) == (2, 3)
        assert isinstance(op, _YamlDataProto)
        assert op.get_yaml_data() == fake_entry

        # 3. broadcast
        assert op.dispatch_infer_shape(d1, inputs=[d2]) == (2, 2)

        # 4. no shapes
        assert op.dispatch_infer_shape(Dummy()) == ()

        # 5. Broadcast mismatch raises ValueError
        d3 = Dummy()
        pytest.MonkeyPatch().setattr(d3, "shape", (3, 3), raising=False)
        d4 = Dummy()
        pytest.MonkeyPatch().setattr(d4, "shape", (2, 2), raising=False)
        with pytest.raises(ValueError):
            op.dispatch_infer_shape(d3, d4)

    finally:
        if original is None:
            del _YAML_REGISTRY["FakeShapeOp"]
        else:
            _YAML_REGISTRY["FakeShapeOp"] = original
        if "FakeShapeOp" in _OP_REGISTRY:
            del _OP_REGISTRY["FakeShapeOp"]


def test_get_all_ops() -> None:
    """Test function test_get_all_ops."""
    from ml_switcheroo_compiler.ops.registry import get_all_ops

    ops = get_all_ops()
    assert len(ops) > 0


def test_get_op_exceptions() -> None:
    """Test function test_get_op_exceptions."""
    with pytest.raises(KeyError):
        get_op("NonExistentOp")
    with pytest.raises(KeyError):
        get_op("TotallyMissingOpXYZ")


def test_registry_shim_methods() -> None:
    """Test function test_registry_shim_methods."""
    from ml_switcheroo_compiler.ops.registry import (
        _RegistryShim,
        _RegistryShimFix,
        get_backend_mapping,
        get_frontend,
        register_frontend,
    )

    data = {"Op1": {"variants": {"np": {"generator": "np_gen", "eager": "np_eager"}}}}

    # _RegistryShim
    shim1 = _RegistryShim(data)
    assert shim1.get_generator_mapping("np", "Op1") == "np_gen"
    assert shim1.get_generator_mapping("np", "Op2") is None

    # _RegistryShimFix
    shim2 = _RegistryShimFix(data)
    assert shim2.get_generator_mapping("np", "Op1") == "np_gen"
    assert shim2.get_generator_mapping("np", "Op2") is None

    assert shim2.get_eager_mapping("np", "Op1") == "np_eager"
    assert shim2.get_eager_mapping("np", "Op2") is None

    assert shim2.get_op("Op1") == data["Op1"]

    # get_backend_mapping
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY

    _YAML_REGISTRY["Op1"] = data["Op1"]
    assert get_backend_mapping("Op1") == data["Op1"]["variants"]
    assert get_backend_mapping("Op2") == {}

    # frontends
    @register_frontend("TestFrontend")
    class TestFE:
        """Test class TestFE."""

        pass

    assert get_frontend("TestFrontend") is TestFE
    with pytest.raises(KeyError):
        get_frontend("MissingFE")


def test_register_op_duplicate() -> None:
    """Test function test_register_op_duplicate."""
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.registry import register_op

    @register_op("TestDupOp")
    class TestDup1(OpDef):
        """Test class TestDup1."""

        pass

    with pytest.raises(ValueError):

        @register_op("TestDupOp")
        class TestDup2(OpDef):
            """Test class TestDup2."""

            pass

    _OP_REGISTRY.pop("TestDupOp", None)


def test_util_registry() -> None:
    """Test function test_util_registry."""
    from ml_switcheroo_compiler.ops.registry import get_util, register_util

    @register_util("TestUtil")
    def my_util() -> None:
        """Test function my_util."""
        pass

    assert get_util("TestUtil") is my_util
    with pytest.raises(KeyError):
        get_util("MissingUtil")


def test_yaml_registry_missing_file() -> None:
    """Test function test_yaml_registry_missing_file."""
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY, _load_yaml_registry

    old_registry = _YAML_REGISTRY.copy() if _YAML_REGISTRY else {}
    try:
        with patch("os.path.exists", return_value=False):
            _load_yaml_registry(force=True)
    finally:
        _YAML_REGISTRY.clear()
        _YAML_REGISTRY.update(old_registry)
    assert _YAML_REGISTRY == old_registry
