from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.mapping_loader import _MAPPING_CACHE, load_backend_mappings, resolve_target_api


def test_load_backend_mappings():
    # Clear cache
    _MAPPING_CACHE.clear()

    # Test valid
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", MagicMock()):
            with patch("yaml.safe_load", return_value={"backend_name": "test_bk", "operations": {}}):
                schema = load_backend_mappings("test_bk")
                assert schema.backend_name == "test_bk"

                # Test cache
                schema2 = load_backend_mappings("test_bk")
                assert schema2 is schema


def test_load_backend_mappings_missing():
    _MAPPING_CACHE.clear()
    with patch("os.path.exists", return_value=False):
        schema = load_backend_mappings("missing_bk")
        assert schema.backend_name == "missing_bk"
        assert len(schema.operations) == 0


def test_resolve_target_api():
    # custom_op
    res = resolve_target_api("custom_op", custom_code="lambda x: x * 2")
    assert res(5) == 10

    # custom_op error
    res_err = resolve_target_api("custom_op", custom_code="lambda x: x * y")  # y undefined
    assert res_err is None or isinstance(res_err, type(lambda x: x))  # In try block, evaluating lambda works, error is runtime

    res_err_eval = resolve_target_api("custom_op", custom_code="invalid syntax +")
    assert res_err_eval is None

    # module import
    res2 = resolve_target_api("math.sqrt")
    import math

    assert res2 == math.sqrt

    # backend module fallback
    class MockBackend:
        def my_func(self):
            return 5

    res3 = resolve_target_api("my_func", backend_module=MockBackend())
    assert res3.__name__ == "my_func"

    # missing module
    res4 = resolve_target_api("missing_module.func")
    assert res4 is None

    # Empty
    assert resolve_target_api("") is None


def test_resolve_target_api_with_backend_globals():
    class MockBackend:
        MY_GLOBAL = 10

    res = resolve_target_api("custom_op", custom_code="lambda x: x + MY_GLOBAL", backend_module=MockBackend())
    assert res(5) == 15
