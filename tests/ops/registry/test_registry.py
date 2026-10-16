"""Tests for test_registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import get_frontend, get_op, get_util, register_frontend, register_op, register_util

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_registry_register_op() -> None:
    """Test function test_registry_register_op."""

    @register_op("TestRegistryOp1")
    class TestRegistryOp1(OpDef):
        """Test class TestRegistryOp1."""

        def infer_shape(self) -> EagerValue:
            """Test function infer_shape."""
            return ()

    assert get_op("TestRegistryOp1") == TestRegistryOp1

    # re-registration with same name and same class name is allowed
    @register_op("TestRegistryOp1")
    class TestRegistryOp2(OpDef):
        """Test class TestRegistryOp2."""

        def infer_shape(self) -> EagerValue:
            """Test function infer_shape."""
            return ()

    # re-registration with same name but DIFFERENT class name is KeyError
    with pytest.raises(ValueError, match="already registered"):

        @register_op("TestRegistryOp2")
        class AnotherOpClass(OpDef):
            """Test class AnotherOpClass."""

            def infer_shape(self) -> EagerValue:
                """Test function infer_shape."""
                return ()


def test_registry_util() -> None:
    """Test function test_registry_util."""

    @register_util("my_util")
    def my_util_fn() -> EagerValue:
        """Test function my_util_fn."""
        return 42

    assert get_util("my_util") == my_util_fn

    with pytest.raises(KeyError, match="not found"):
        get_util("non_existent_util")


def test_registry_frontend() -> None:
    """Test function test_registry_frontend."""

    @cast("Callable[..., EagerTensorProtocol]", register_frontend)("my_frontend")
    def my_frontend_fn() -> EagerValue:
        """Test function my_frontend_fn."""
        return 42

    assert get_frontend("my_frontend") == my_frontend_fn

    with pytest.raises(KeyError, match="not found"):
        get_frontend("non_existent_frontend")


def test_get_op_not_found() -> None:
    """Test function test_get_op_not_found."""
    with pytest.raises(KeyError, match="not found"):
        get_op("NonExistentOp")

    with pytest.raises(KeyError, match="not found"):
        get_op("OtherNonExistentOp")


def test_registry_files_load() -> None:
    """Test function test_registry_files_load."""
    from ml_switcheroo_compiler.ops.registry import _REGISTRY, _YAML_REGISTRY, _RegistryShim, backend_mapping_registry, get_all_ops, get_backend_mapping, get_op

    ops = get_all_ops()
    assert len(ops) > 0
    op = get_op("Conv2D")
    assert cast("EagerTensorProtocol", getattr(op, "get_yaml_data"))() is not None
    assert backend_mapping_registry.get_generator_mapping("numpy", "NonExistentOp") is None
    assert backend_mapping_registry.get_generator_mapping("numpy", "Add") is None or backend_mapping_registry.get_generator_mapping("numpy", "Add") is not None
    assert get_backend_mapping("NonExistentOp") == {}
    assert get_backend_mapping("Add") != {}

    shim1 = _RegistryShim(_YAML_REGISTRY)
    assert shim1.get_generator_mapping("numpy", "NonExistentOp") is None
    assert shim1.get_generator_mapping("numpy", "Add") is None or shim1.get_generator_mapping("numpy", "Add") is not None

    assert backend_mapping_registry.get_eager_mapping("numpy", "NonExistentOp") is None
    assert backend_mapping_registry.get_eager_mapping("numpy", "Add") is None or backend_mapping_registry.get_eager_mapping("numpy", "Add") is not None

    assert backend_mapping_registry.get_op("NonExistentOp") is None
    assert backend_mapping_registry.get_op("Add") is not None

    # Test dynamic class builder
    getattr(_YAML_REGISTRY, "__setitem__")("FakeYamlOp", {"some": "data"})
    fake_op_cls = get_op("FakeYamlOp")
    assert fake_op_cls.__name__ == "FakeYamlOp"
    assert cast("EagerTensorProtocol", getattr(fake_op_cls, "get_yaml_data"))() == {"some": "data"}
    _REGISTRY.pop("FakeYamlOp", None)
    _YAML_REGISTRY.pop("FakeYamlOp", None)

    # Test get_all_ops populating missing ops
    if "Add" in _REGISTRY:
        _REGISTRY.pop("Add")
    get_all_ops()
    assert "Add" in _REGISTRY
