"""Tests for backend YAML mapping loader and dynamic eager dispatch."""

from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.mapping_loader import (
    _MAPPING_CACHE,
    BackendMappingSchema,
    KwargTranslation,
    OpMappingSchema,
    _load_yaml_dir,
    _read_and_merge,
    _resolve_by_import,
    _resolve_custom_code,
    _resolve_from_backend_module,
    dispatch_eager_op,
    load_backend_mappings,
    resolve_target_api,
    translate_kwargs,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def test_kwarg_translation_model() -> None:
    """Test KwargTranslation model initialization."""
    kt = KwargTranslation(target_name="axis", default_value=0)
    assert kt.target_name == "axis"
    assert kt.default_value == 0


def test_load_backend_mappings_cache_and_paths() -> None:
    """Test load_backend_mappings with cache, yaml_path, and eager_yaml_path."""
    _MAPPING_CACHE.clear()

    # Both mappings.yaml and eager_mappings.yaml exist
    def fake_exists(p: str) -> bool:
        return True

    mock_yaml_data = {"operations": {"add": {"target_api": "math.add"}}}
    with patch("os.path.exists", side_effect=fake_exists):
        with patch("builtins.open", MagicMock()):
            with patch("yaml.safe_load", return_value=mock_yaml_data):
                schema1 = load_backend_mappings("fake_backend")
                assert schema1.backend_name == "fake_backend"
                assert "add" in schema1.operations

                # Cache hit
                schema2 = load_backend_mappings("fake_backend")
                assert schema2 is schema1


def test_load_backend_mappings_missing_files() -> None:
    """Test load_backend_mappings when files do not exist."""
    _MAPPING_CACHE.clear()
    with patch("os.path.exists", return_value=False):
        schema = load_backend_mappings("nonexistent_bk")
        assert schema.backend_name == "nonexistent_bk"
        assert len(schema.operations) == 0


def test_read_and_merge_variations(tmp_path) -> None:
    """Test _read_and_merge with different YAML structures."""
    # Dict with operations key
    f1 = tmp_path / "f1.yaml"
    f1.write_text("operations:\n  op1:\n    target_api: a.b\n", encoding="utf-8")
    t1 = {}
    _read_and_merge(str(f1), t1)
    assert "op1" in t1

    # Dict without operations key
    f2 = tmp_path / "f2.yaml"
    f2.write_text("op2:\n  target_api: c.d\n", encoding="utf-8")
    t2 = {}
    _read_and_merge(str(f2), t2)
    assert "op2" in t2

    # Non-dict data (e.g. list)
    f3 = tmp_path / "f3.yaml"
    f3.write_text("- item1\n- item2\n", encoding="utf-8")
    t3 = {}
    _read_and_merge(str(f3), t3)
    assert t3 == {}


def test_load_yaml_dir(tmp_path) -> None:
    """Test _load_yaml_dir with directory of yaml and non-yaml files."""
    d = tmp_path / "yamls"
    d.mkdir()
    (d / "ignored.txt").write_text("hello", encoding="utf-8")
    (d / "valid.yaml").write_text("opX: {target_api: x.y}\n", encoding="utf-8")

    res = {}
    _load_yaml_dir(str(d), res)
    assert "opX" in res

    # Non-directory
    empty_res = {}
    _load_yaml_dir(str(tmp_path / "not_a_dir"), empty_res)
    assert empty_res == {}


def test_resolve_custom_code() -> None:
    """Test _resolve_custom_code branches."""
    res = _resolve_custom_code("lambda x: x * 2", None)
    assert res(5) == 10

    # With backend_module having globals and property raising error
    class MockBackend:
        MY_VAR = 7

        @property
        def err(self):
            raise RuntimeError("Property access failed")

    res2 = _resolve_custom_code("lambda x: x + MY_VAR", MockBackend())
    assert res2(3) == 10

    # Invalid code
    assert _resolve_custom_code("lambda invalid syntax", None) is None


def test_resolve_by_import() -> None:
    """Test _resolve_by_import success and failure."""
    import math

    assert _resolve_by_import("math.sqrt") is math.sqrt
    assert _resolve_by_import("non_existent_pkg.sub.func") is None


def test_resolve_from_backend_module() -> None:
    """Test _resolve_from_backend_module paths and branches."""

    class SubMod:
        def my_target(self) -> int:
            return 42

    class ParentMod:
        def __init__(self) -> None:
            self.sub = SubMod()
            self.direct_target = lambda: 100

    mod = ParentMod()

    # Successfully traverse sub_parts (line 169)
    res1 = _resolve_from_backend_module("torch.sub.my_target", mod)
    assert callable(res1)
    assert res1() == 42

    # Miss sub_parts, but backend_module has parts[-1] (line 166 branch 1)
    res2 = _resolve_from_backend_module("torch.missing.direct_target", mod)
    assert callable(res2)
    assert res2() == 100

    # Miss sub_parts, and backend_module does NOT have parts[-1] (line 166 branch 2)
    res3 = _resolve_from_backend_module("torch.missing.nonexistent", mod)
    assert res3 is None

    # Single-part api_str to exercise else: sub_parts = parts (line 163)
    res_single = _resolve_from_backend_module("direct_target", mod)
    assert callable(res_single)
    assert res_single() == 100

    # dask.array.target to exercise sub_parts = parts[2:]
    class MockDaskMod:
        def my_sum(self) -> str:
            return "dask_sum"

    dask_mod = MockDaskMod()
    res_dask = _resolve_from_backend_module("dask.array.my_sum", dask_mod)
    assert callable(res_dask)
    assert res_dask() == "dask_sum"


def test_resolve_target_api() -> None:
    """Test resolve_target_api paths and branches."""
    # custom_op
    res = resolve_target_api("custom_op", custom_code="lambda x: x * 2")
    assert res(5) == 10

    # Empty api_str
    assert resolve_target_api("") is None

    # backend_module is None, succeeds via import (line 205)
    import math

    assert resolve_target_api("math.sqrt", backend_module=None) is math.sqrt

    # backend_module is None, fails import -> returns None (line 200 backend_module is None)
    assert resolve_target_api("unresolvable_module_xyz.func") is None

    # backend_module has dotted attribute directly via setattr (line 200 with backend_module)
    dummy = type("Dummy", (), {})()
    dummy_func = lambda: 777
    setattr(dummy, "special.pkg.func", dummy_func)
    res_dotted = resolve_target_api("special.pkg.func", backend_module=dummy)
    assert res_dotted == dummy_func

    # backend_module has last part
    class MockBackend:
        def my_func(self) -> int:
            return 5

    res3 = resolve_target_api("some.prefix.my_func", backend_module=MockBackend())
    assert res3.__name__ == "my_func"

    # backend_module has nested structure without root attribute (lines 198-200)
    class InnerMod:
        def target(self) -> int:
            return 999

    class OuterMod:
        def __init__(self) -> None:
            self.inner = InnerMod()

    res_mod_res = resolve_target_api("inner.target", backend_module=OuterMod())
    assert callable(res_mod_res)
    assert res_mod_res() == 999

    # Direct import
    import math

    assert resolve_target_api("math.sqrt") == math.sqrt


def test_translate_kwargs() -> None:
    """Test translate_kwargs mapping."""
    res = translate_kwargs({"dim": "axis"}, {"dim": 1, "keepdim": False})
    assert res == {"axis": 1, "keepdim": False}


def test_dispatch_eager_op() -> None:
    """Test dispatch_eager_op with all branches including default_kwargs and is_method."""
    schema = BackendMappingSchema(
        backend_name="test_backend_dispatch",
        operations={
            "OpWithDefaults": OpMappingSchema(
                target_api="std_func",
                kwarg_translations={"dim": "axis"},
                default_kwargs={"axis": 0, "keepdims": True},
            ),
            "MethodOp": OpMappingSchema(
                target_api="do_method",
                is_method=True,
            ),
            "MethodOpFallback": OpMappingSchema(
                target_api="missing_method",
                is_method=True,
            ),
            "UnresolvableOp": OpMappingSchema(
                target_api="unresolvable.path.xyz",
            ),
        },
    )
    _MAPPING_CACHE["test_backend_dispatch"] = schema

    # Op not in schema
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        dispatch_eager_op("test_backend_dispatch", "UnknownOp", [], {})

    # Op cannot be resolved
    with pytest.raises(BackendNotSupportedError, match="could not be resolved"):
        dispatch_eager_op("test_backend_dispatch", "UnresolvableOp", [], {})

    # OpWithDefaults: test default_kwargs branches (one overwritten, one defaulted)
    mock_mod = MagicMock()
    mock_mod.std_func.side_effect = lambda *args, **kwargs: kwargs
    # Caller passes dim=2 -> translated to axis=2. keepdims is defaulted to True!
    # Tests def_k in translated_kwargs (False for keepdims -> lines 261-262, True for axis)
    res_kwargs = dispatch_eager_op(
        "test_backend_dispatch",
        "OpWithDefaults",
        [],
        {"dim": 2},
        backend_module=mock_mod,
    )
    assert res_kwargs == {"axis": 2, "keepdims": True}

    # is_method True and method exists
    class CallableObj:
        def do_method(self, val: int) -> int:
            return val * 10

    c_obj = CallableObj()
    res_m = dispatch_eager_op(
        "test_backend_dispatch",
        "MethodOp",
        [c_obj, 5],
        {},
        backend_module=c_obj,
    )
    assert res_m == 50

    # is_method True but method missing or not callable -> falls back to func
    mock_fallback_mod = MagicMock()
    mock_fallback_mod.missing_method.return_value = "fallback_called"
    res_fb = dispatch_eager_op(
        "test_backend_dispatch",
        "MethodOpFallback",
        [object()],
        {},
        backend_module=mock_fallback_mod,
    )
    assert res_fb == "fallback_called"
