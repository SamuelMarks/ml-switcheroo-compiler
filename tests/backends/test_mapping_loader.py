from unittest.mock import MagicMock, mock_open, patch

from ml_switcheroo_compiler.backends.mapping_loader import _MAPPING_CACHE, load_backend_mappings, resolve_target_api


def test_load_backend_mappings_missing(tmp_path):
    with patch("os.path.dirname", return_value=str(tmp_path)):
        _MAPPING_CACHE.clear()
        schema = load_backend_mappings("fake_backend")
        assert schema.backend_name == "fake_backend"
        assert len(schema.operations) == 0


def test_load_backend_mappings_existing():
    _MAPPING_CACHE.clear()
    yaml_data = """
backend_name: "test_backend"
operations:
  TestOp:
    target_api: "math.sin"
    is_method: false
    kwarg_translations: {"x": "y"}
    supported_dtypes: ["float32"]
"""
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml_data)):
            schema = load_backend_mappings("test_backend")
            assert schema.backend_name == "test_backend"
            assert "TestOp" in schema.operations
            assert schema.operations["TestOp"].target_api == "math.sin"

            # test cache
            schema2 = load_backend_mappings("test_backend")
            assert schema2 is schema


class _FakeBackend:
    def some_func(self, x):
        return x * 2


def test_resolve_target_api():
    # Empty
    assert resolve_target_api("") is None

    # Import string
    assert resolve_target_api("math.sin").__name__ == "sin"

    # Custom eval code
    mock_module = _FakeBackend()

    res = resolve_target_api("custom_op", "backend_module.some_func(3)", mock_module)
    assert res == 6

    # Custom eval exception (method does not exist on class)
    assert resolve_target_api("custom_op", "backend_module.does_not_exist_throw()", mock_module) is None

    # Target API attribute on backend_module
    mock_backend = MagicMock()
    mock_backend.my_func = "hello"
    assert resolve_target_api("my_func", backend_module=mock_backend) == "hello"

    # Target API missing
    assert resolve_target_api("non_existent_module.func") is None

    # Test error in import resolution
    assert resolve_target_api("os.non_existent_func") is None
