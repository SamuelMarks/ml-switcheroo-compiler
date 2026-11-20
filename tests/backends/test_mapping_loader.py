"""Tests for backend YAML mapping loader and dynamic eager dispatch."""

from __future__ import annotations

import pathlib
import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
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
from tests.testing_utils import MockTestStub, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


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
        """Test function fake_exists.

        Args:
            p: Test parameter `p`.

        Returns:
            bool: Computed output value from `fake_exists`.
        """
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


def test_read_and_merge_variations(tmp_path: Path) -> None:
    """Test _read_and_merge with different YAML structures.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    # Dict with operations key
    f1 = tmp_path / "f1.yaml"
    f1.write_text("operations:\n  op1:\n    target_api: a.b\n", encoding="utf-8")
    t1: dict[str, BackendArg] = {}
    _read_and_merge(str(f1), t1)
    assert "op1" in t1

    # Dict without operations key
    f2 = tmp_path / "f2.yaml"
    f2.write_text("op2:\n  target_api: c.d\n", encoding="utf-8")
    t2: dict[str, BackendArg] = {}
    _read_and_merge(str(f2), t2)
    assert "op2" in t2

    # Non-dict data (e.g. list)
    f3 = tmp_path / "f3.yaml"
    f3.write_text("- item1\n- item2\n", encoding="utf-8")
    t3: dict[str, BackendArg] = {}
    _read_and_merge(str(f3), t3)
    assert t3 == {}


def test_load_yaml_dir(tmp_path: Path) -> None:
    """Test _load_yaml_dir with directory of yaml and non-yaml files.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    d = tmp_path / "yamls"
    d.mkdir()
    (d / "ignored.txt").write_text("hello", encoding="utf-8")
    (d / "valid.yaml").write_text("opX: {target_api: x.y}\n", encoding="utf-8")

    res: dict[str, BackendArg] = {}
    _load_yaml_dir(str(d), res)
    assert "opX" in res

    # Non-directory
    empty_res: dict[str, BackendArg] = {}
    _load_yaml_dir(str(tmp_path / "not_a_dir"), empty_res)
    assert empty_res == {}


def test_resolve_custom_code() -> None:
    """Test _resolve_custom_code branches."""
    res = _resolve_custom_code("lambda x: x * 2", None)
    assert require_callable(res)(5) == 10

    # With backend_module having globals and property raising error
    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        MY_VAR = 7

        @property
        def err(self) -> None:
            """Test function err.

            Raises:
                RuntimeError: When raised during `err` (Property access failed).
            """
            raise RuntimeError("Property access failed")

    res2 = _resolve_custom_code("lambda x: x + MY_VAR", MockBackend())
    assert require_callable(res2)(3) == 10

    # Invalid code
    assert _resolve_custom_code("lambda invalid syntax", None) is None

    # Comprehensive _safe_expr_eval AST coverage (varargs, kwargs, defaults, slices, comprehensions, operators,
    # security guards)
    from ml_switcheroo_compiler.backends._safe_expr_eval import evaluate_safe_expression

    fn_full = evaluate_safe_expression(
        "lambda a, b=2, *xs, **kw: [a + b + len(xs) + kw.get('c', 0) for i in range(2)]", {}
    )
    assert require_callable(fn_full)(1) == [3, 3]
    assert require_callable(fn_full)(a=1, b=4, c=5) == [10, 10]
    assert require_callable(fn_full)(1, 2, 9, 8) == [5, 5]

    fn_plain = evaluate_safe_expression(
        "lambda x: (x[0:2], [1, 2], +x[0], -x[0], not False, True and False, False or True, 1 if x[0] == 1 else 0)",
        {},
    )
    assert require_callable(fn_plain)([1, 2, 3]) == ([1, 2], [1, 2], 1, -1, True, False, True, 1)
    assert require_callable(fn_plain)([5, 2, 3])[-1] == 0

    fn_ops = evaluate_safe_expression(
        "lambda a, b: (a - b, a * b, a / b, a // b, a % b, a ** b, a != b, a < b, a <= b, a > b, a >= b)", {}
    )
    assert require_callable(fn_ops)(6, 2) == (4, 12, 3.0, 3, 0, 36, True, False, False, True, True)

    fn_star = evaluate_safe_expression("lambda f, xs, kw: f(*xs, s=1, **kw)", {})
    assert require_callable(fn_star)(lambda p, q, r=0, s=0: p + q + r + s, [3, 4], {"r": 5}) == 13

    fn_gen = evaluate_safe_expression("lambda n: tuple(i * 2 for i in range(n))", {})
    assert require_callable(fn_gen)(3) == (0, 2, 4)

    fn_builtins = evaluate_safe_expression(
        (
            "lambda x: (bool(x), int(x), float(x), str(x), abs(-x), list((1, 2)), sum([1, 2, 3]), max([1,"
            " 5, 2]), min(4, 2, 9), list(range(1, 5, 2)), slice(1, 3))"
        ),
        {},
    )
    assert require_callable(fn_builtins)(3) == (
        True,
        3,
        3.0,
        "3",
        3,
        [1, 2],
        6,
        5,
        2,
        [1, 3],
        require_callable(slice)(1, 3),
    )

    class _PubHolder:
        """Public attribute holder for safe hasattr/getattr verification."""

        val = 11

    holder = _PubHolder()
    fn_attr = evaluate_safe_expression("lambda h: h.val if hasattr(h, 'val') and not hasattr(h, '_secret') else 0", {})
    assert require_callable(fn_attr)(holder) == 11

    with pytest.raises(TypeError, match="Missing required argument"):
        require_callable(fn_plain)()
    with pytest.raises(TypeError, match="Too many positional arguments"):
        require_callable(fn_plain)([1], [2])
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        require_callable(fn_plain)([1], extra=9)
    with pytest.raises(NameError, match="Disallowed private identifier"):
        evaluate_safe_expression("_secret", {})
    with pytest.raises(NameError, match="Unbound identifier"):
        evaluate_safe_expression("unknown_symbol", {})
    with pytest.raises(ValueError, match="Disallowed private attribute"):
        require_callable(evaluate_safe_expression)("lambda h: h._priv", {})(holder)
    with pytest.raises(ValueError, match="Unsupported binary operator"):
        evaluate_safe_expression("1 << 2", {})
    with pytest.raises(ValueError, match="Unsupported unary operator"):
        evaluate_safe_expression("~1", {})
    with pytest.raises(ValueError, match="Unsupported comparison operator"):
        evaluate_safe_expression("1 is 1", {})
    with pytest.raises(ValueError, match="Unsupported comprehension structure"):
        evaluate_safe_expression("[x for x in [1, 2] if x > 1]", {})
    with pytest.raises(ValueError, match="Unsupported AST expression node"):
        evaluate_safe_expression("{1: 2}", {})


def test_resolve_by_import() -> None:
    """Test _resolve_by_import success, nested submodule traversal, and failure."""
    import math

    resolved_cls = _resolve_by_import("http.cookiejar.CookieJar")
    assert resolved_cls is not None
    assert _resolve_by_import("math.sqrt") is math.sqrt
    assert _resolve_by_import("math.missing_submod.func") is None
    assert _resolve_by_import("non_existent_pkg.sub.func") is None


def test_resolve_from_backend_module() -> None:
    """Test _resolve_from_backend_module paths and branches."""

    class SubMod:
        """Test class SubMod."""

        def my_target(self) -> int:
            """Test function my_target.

            Returns:
                int: Computed output value from `my_target`.
            """
            return 42

    class ParentMod(MockTestStub):
        """Test class ParentMod."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.sub = SubMod()
            self.direct_target = lambda: 100

    mod = ParentMod()

    # Successfully traverse sub_parts (line 169)
    res1 = _resolve_from_backend_module("torch.sub.my_target", mod)
    assert require_callable(res1)() == 42

    # Miss sub_parts, but backend_module has parts[-1] (line 166 branch 1)
    res2 = _resolve_from_backend_module("torch.missing.direct_target", mod)
    assert require_callable(res2)() == 100

    # Miss sub_parts, and backend_module does NOT have parts[-1] (line 166 branch 2)
    res3 = _resolve_from_backend_module("torch.missing.nonexistent", mod)
    assert res3 is None

    # Single-part api_str to exercise else: sub_parts = parts (line 163)
    res_single = _resolve_from_backend_module("direct_target", mod)
    assert require_callable(res_single)() == 100

    # dask.array.target to exercise sub_parts = parts[2:]
    class MockDaskMod(MockTestStub):
        """Test class MockDaskMod."""

        def my_sum(self) -> str:
            """Test function my_sum.

            Returns:
                str: Computed output value from `my_sum`.
            """
            return "dask_sum"

    dask_mod = MockDaskMod()
    res_dask = _resolve_from_backend_module("dask.array.my_sum", dask_mod)
    assert require_callable(res_dask)() == "dask_sum"


def test_resolve_target_api() -> None:
    """Test resolve_target_api paths and branches."""
    # custom_op
    res = resolve_target_api("custom_op", custom_code="lambda x: x * 2")
    assert require_callable(res)(5) == 10

    # Empty api_str
    assert resolve_target_api("") is None

    # backend_module is None, succeeds via import (line 205)
    import math

    assert resolve_target_api("math.sqrt", backend_module=None) is math.sqrt

    # backend_module is None, fails import -> returns None (line 200 backend_module is None)
    assert resolve_target_api("unresolvable_module_xyz.func") is None

    # backend_module has dotted attribute directly via setattr (line 200 with backend_module)
    def dummy_func() -> BackendArg:
        """Return dummy constant 777.

        Returns:
            BackendArg: Computed output value from `dummy_func`.
        """
        return 777

    pkg_ns = type("PkgNs", (), {"func": staticmethod(dummy_func)})()
    special_ns = type("SpecialNs", (), {"pkg": pkg_ns})()
    dummy = type("Dummy", (), {"special": special_ns})()
    res_dotted = resolve_target_api("special.pkg.func", backend_module=dummy)
    assert res_dotted == dummy_func

    # backend_module has last part
    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        def my_func(self) -> int:
            """Test function my_func.

            Returns:
                int: Computed output value from `my_func`.
            """
            return 5

    res3 = resolve_target_api("some.prefix.my_func", backend_module=MockBackend())
    assert require_callable(res3)() == 5

    # backend_module has nested structure without root attribute (lines 198-200)
    class InnerMod:
        """Test class InnerMod."""

        def target(self) -> int:
            """Test function target.

            Returns:
                int: Computed output value from `target`.
            """
            return 999

    class OuterMod(MockTestStub):
        """Test class OuterMod."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.inner = InnerMod()

    res_mod_res = resolve_target_api("inner.target", backend_module=OuterMod())
    assert require_callable(res_mod_res)() == 999

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
                kwarg_map={"source_dim": "mapped_dim", "ignore_none": None},
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
        {"dim": 2, "source_dim": 5},
        backend_module=mock_mod,
    )
    assert res_kwargs == {"axis": 2, "mapped_dim": 5, "keepdims": True}

    # is_method True and method exists
    class CallableObj(MockTestStub):
        """Test class CallableObj."""

        def do_method(self, val: int) -> int:
            """Test function do_method.

            Args:
                val: Test parameter `val`.

            Returns:
                int: Computed output value from `do_method`.
            """
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

    class _BareDummy(MockTestStub):
        """Empty dummy class without attributes."""

    # is_method True but method missing or not callable -> falls back to func
    mock_fallback_mod = MagicMock()
    mock_fallback_mod.missing_method.return_value = "fallback_called"
    res_fb = dispatch_eager_op(
        "test_backend_dispatch",
        "MethodOpFallback",
        [_BareDummy()],
        {},
        backend_module=mock_fallback_mod,
    )
    assert res_fb == "fallback_called"


def test_read_and_merge_with_operation(tmp_path: pathlib.Path) -> None:
    """Test _read_and_merge when file contains top-level operation key.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    yaml_file = tmp_path / "op.yaml"
    yaml_file.write_text("operation: test_single_op\ntarget_api: foo.bar\n")
    target: dict[str, BackendArg] = {}
    _read_and_merge(str(yaml_file), target)
    assert "test_single_op" in target
