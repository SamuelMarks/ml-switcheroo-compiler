"""Tests for the operations registry."""

from ml_switcheroo_compiler.ops.registry import backend_mapping_registry as registry


def test_registry_get_op():
    """Test getting an operation."""
    # Transpose should definitely exist.
    op = registry.get_op("Transpose")
    assert op is not None
    assert op["description"] == "The Transpose operation."

    # Missing op
    assert registry.get_op("NonExistentOp123") is None


def test_registry_get_eager_mapping():
    """Test getting eager mapping."""
    # Cupy has eager for Transpose
    mapping = registry.get_eager_mapping("cupy", "Transpose")
    assert mapping == "cp.transpose"

    # Missing backend
    assert registry.get_eager_mapping("nonexistent_backend", "transpose") is None

    # Missing op
    assert registry.get_eager_mapping("cupy", "NonExistentOp123") is None


def test_registry_get_generator_mapping():
    """Test getting generator mapping."""
    mapping = registry.get_generator_mapping("numpy", "Transpose")
    assert mapping == "np.transpose"

    # Missing backend
    assert registry.get_generator_mapping("nonexistent_backend", "transpose") is None

    # Missing op
    assert registry.get_generator_mapping("numpy", "NonExistentOp123") is None


def test_registry_extras():
    """test_registry_extras."""
    import pytest

    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY, backend_mapping_registry, get_all_ops, get_backend_mapping, get_frontend, get_op, register_frontend

    with pytest.raises(KeyError):
        get_op("NonExistentOp")
    with pytest.raises(ValueError):
        get_op("SomeOtherMissingOpThatIsNotNonExistentOp")

    ops = get_all_ops()
    assert isinstance(ops, dict)

    _YAML_REGISTRY["DummyOpForTest"] = {"variants": {"test_backend": {"generator": "test_gen"}}}

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


def test_registry_remaining():
    """test_registry_remaining."""
    import pytest

    from ml_switcheroo_compiler.ops.registry import get_op, get_util, register_op, register_util

    @register_op("TestDuplicate")
    class A:
        """A."""

        pass

    with pytest.raises(ValueError):

        @register_op("TestDuplicate")
        class B:
            """B."""

            pass

    @register_util("MyUtil")
    def my_util():
        """my_util."""
        pass

    assert get_util("MyUtil") == my_util
    with pytest.raises(KeyError):
        get_util("MissingUtil")

    # hit 112
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY

    _YAML_REGISTRY["DynamicTestOp123"] = {"test": 1}
    op_cls = get_op("DynamicTestOp123")
    assert op_cls.get_yaml_data() == {"test": 1}

    # hit 170-175 inside _RegistryShim
    class DummyShim:
        """DummyShim."""

        def __init__(self, data):
            """__init__."""
            self.operations = data

        def get_generator_mapping(self, prefix, op_name):
            """get_generator_mapping."""
            op = self.operations.get(op_name, {})
            if not op:
                return None
            variants = op.get("variants", {})
            backend = variants.get(prefix, {})
            return backend.get("generator")

    from ml_switcheroo_compiler.ops.registry import _RegistryShim

    shim = _RegistryShim({"TestOpShim": {"variants": {"test_backend": {"generator": "test_gen_shim"}}}})
    assert shim.get_generator_mapping("test_backend", "TestOpShim") == "test_gen_shim"
    assert shim.get_generator_mapping("test_backend", "MissingOpShim") is None


def test_registry_yaml_missing():
    """test_registry_yaml_missing."""
    import copy

    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY, _load_yaml_registry

    orig = copy.deepcopy(_YAML_REGISTRY)
    _YAML_REGISTRY.clear()

    from unittest.mock import patch

    with patch("os.path.exists", return_value=False):
        _load_yaml_registry(force=True)
        assert not _YAML_REGISTRY

    _YAML_REGISTRY.update(orig)
