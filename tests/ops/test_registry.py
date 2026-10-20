"""Tests for test_registry."""

from __future__ import annotations

import typing
from unittest import mock
from unittest.mock import patch

from ml_switcheroo_compiler.ops.registry import backend_mapping_registry as registry
from ml_switcheroo_compiler.ops.registry import get_all_ops, get_op

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_registry_get_op() -> None:
    """Test getting an operation."""
    # Transpose should definitely exist.
    op = registry.get_op("Transpose")
    assert op is not None
    assert typing.cast("EagerTensorProtocol", op["description"]) == typing.cast("EagerTensorProtocol", "The Transpose operation.")

    # Missing op
    assert registry.get_op("NonExistentOp123") is None


def test_registry_get_eager_mapping() -> None:
    """Test getting eager mapping."""
    # Cupy has eager for Transpose
    mapping = registry.get_eager_mapping("cupy", "Transpose")
    assert mapping == "cp.transpose"

    # Missing backend
    assert registry.get_eager_mapping("nonexistent_backend", "transpose") is None

    # Missing op
    assert registry.get_eager_mapping("cupy", "NonExistentOp123") is None


def test_registry_get_generator_mapping() -> None:
    """Test getting generator mapping."""
    mapping = registry.get_generator_mapping("numpy", "Transpose")
    assert mapping == "np.transpose"

    # Missing backend
    assert registry.get_generator_mapping("nonexistent_backend", "transpose") is None

    # Missing op
    assert registry.get_generator_mapping("numpy", "NonExistentOp123") is None


def test_registry_extras() -> None:
    """test_registry_extras."""
    import pytest

    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY, backend_mapping_registry, get_all_ops, get_backend_mapping, get_frontend, get_op, register_frontend

    with pytest.raises(KeyError):
        get_op("NonExistentOp")
    with pytest.raises(KeyError):
        get_op("SomeOtherMissingOpThatIsNotNonExistentOp")

    ops = get_all_ops()
    assert isinstance(ops, dict)

    _YAML_REGISTRY["DummyOpForTest"] = {"variants": {"test_backend": {"generator": "test_gen"}}}

    assert id(_YAML_REGISTRY) == id(backend_mapping_registry.operations)

    assert backend_mapping_registry.get_generator_mapping("test_backend", "DummyOpForTest") == "test_gen"
    assert backend_mapping_registry.get_generator_mapping("test_backend", "MissingOp") is None

    mapping = get_backend_mapping("DummyOpForTest")
    assert mapping == {"test_backend": {"generator": "test_gen"}}
    assert get_backend_mapping("MissingOp") == {}

    @register_frontend("test_frontend")
    class TestFrontend:
        """TestFrontend."""

        pass

    assert get_frontend("test_frontend") == TestFrontend
    with pytest.raises(KeyError):
        get_frontend("missing_frontend")


def test_registry_remaining() -> None:
    """test_registry_remaining."""
    import pytest

    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY, get_op, get_util, register_op, register_util

    @register_op("TestDuplicate")
    class A(OpDef):
        """A."""

        pass

    with pytest.raises(ValueError):

        @register_op("TestDuplicate")
        class B(OpDef):
            """B."""

            pass

    _OP_REGISTRY.pop("TestDuplicate", None)

    @register_util("MyUtil")
    def my_util() -> None:
        """my_util."""
        pass

    assert get_util("MyUtil") == my_util
    with pytest.raises(KeyError):
        get_util("MissingUtil")

    # hit 112
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY

    getattr(_YAML_REGISTRY, "__setitem__")("DynamicTestOp123", {"test": 1})
    op_cls = get_op("DynamicTestOp123")
    assert typing.cast("EagerTensorProtocol", getattr(op_cls, "get_yaml_data"))() == {"test": 1}

    # hit 170-175 inside _RegistryShim
    class DummyShim:
        """DummyShim."""

        def __init__(self, data: EagerValue) -> None:
            """__init__.

            Args:
                data: Test parameter `data`.
            """
            self.operations = data

        def get_generator_mapping(self, prefix: EagerValue, op_name: EagerValue) -> EagerValue:
            """get_generator_mapping.

            Args:
                prefix: Test parameter `prefix`.
                op_name: Test parameter `op_name`.
            """
            op = typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(self.operations, "get")))(op_name, {})
            if not op:
                return None
            variants = typing.cast("EagerTensorProtocol", getattr(op, "get"))("variants", {})
            backend = typing.cast("EagerCallableProtocol", typing.cast("EagerTensorProtocol", getattr(variants, "get")))(prefix, {})
            return typing.cast("EagerTensorProtocol", getattr(backend, "get"))("generator")

    from ml_switcheroo_compiler.ops.registry import _RegistryShim

    shim = _RegistryShim({"TestOpShim": {"variants": {"test_backend": {"generator": "test_gen_shim"}}}})
    assert shim.get_generator_mapping("test_backend", "TestOpShim") == "test_gen_shim"
    assert shim.get_generator_mapping("test_backend", "MissingOpShim") is None


def test_ops_registry_branches() -> None:
    """Test get_op already cached in _REGISTRY and get_all_ops error handling."""
    import ml_switcheroo_compiler.ops.registry as reg

    op_cls = get_op("Add")
    cached = get_op("Add")
    assert cached is op_cls
    reg._YAML_REGISTRY["NonMathFakeOp"] = {}
    with mock.patch("ml_switcheroo_compiler.diagnostics.types_registry.is_non_math_type", side_effect=lambda name: name == "NonMathFakeOp"):
        all_ops = get_all_ops()
        assert "Add" in all_ops
    reg._YAML_REGISTRY.pop("NonMathFakeOp", None)


def test_ops_registry_get_all_ops_key_error() -> None:
    """Test get_all_ops handling of KeyError when get_op fails.

    Returns:
        None
    """
    with patch("ml_switcheroo_compiler.ops.registry.get_op", side_effect=KeyError("Missing op")):
        ops = get_all_ops()
        assert isinstance(ops, dict)
