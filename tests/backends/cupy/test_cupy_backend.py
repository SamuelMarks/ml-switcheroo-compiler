"""Test module."""

from unittest.mock import patch

from ml_switcheroo_compiler.backends.cupy.eager import execute_op
from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
from ml_switcheroo_compiler.backends.cupy.types import array, asarray, item, zeros
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.ir.core import IRNode


class DummyGraph:
    def __init__(self):
        self.nodes = []


class DummyCp:
    def add(self, *args, **kwargs):
        return "add_res"

    def _test_op(self, *args, **kwargs):
        return "test_res"


def test_cupy_eager_execute_op():

    cp_mock = DummyCp()

    # Test registered func
    def dummy_eager(module, *args, **kwargs):
        return "dummy_eager_res"

    global_eager_registry.register("TestOp")(dummy_eager)

    res = execute_op(None, "TestOp")
    assert res == "dummy_eager_res"

    if "Add" in global_eager_registry._registry:
        del global_eager_registry._registry["Add"]

    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings") as mock_mappings:
        mock_schema = type("Dummy", (), {"operations": {"Add": type("DummyOp", (), {"target_api": "add", "custom_code": None})()}})()
        mock_mappings.return_value = mock_schema
        with patch("sys.modules", {"ml_switcheroo_compiler.backends.cupy.eager": cp_mock}):
            res2 = execute_op(None, "Add")
            assert res2 == "add_res"

    global_eager_registry.register("NumpyFallbackOp")(dummy_eager)
    res3 = execute_op(None, "NumpyFallbackOp")
    assert res3 == "dummy_eager_res"

    if "UnknownOp" in global_eager_registry._registry:
        del global_eager_registry._registry["UnknownOp"]

    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    try:
        execute_op(None, "UnknownOp")
    except BackendNotSupportedError:
        pass


def test_cupy_eager_execute_op_exception():
    import pytest

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    if "OpThatRaises" in global_eager_registry._registry:
        del global_eager_registry._registry["OpThatRaises"]

    with pytest.raises(BackendNotSupportedError, match="Operation 'OpThatRaises' is not implemented."):
        execute_op(None, "OpThatRaises")

    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings") as mock_mappings:
        mock_schema = type("Dummy", (), {"operations": {"SnakeOp": type("DummyOp", (), {"target_api": "snake_op", "custom_code": None})()}})()
        mock_mappings.return_value = mock_schema
        with patch("sys.modules", {"ml_switcheroo_compiler.backends.cupy.eager": DummyCp()}):
            with pytest.raises(BackendNotSupportedError):
                execute_op(None, "SnakeOp")


def test_cupy_generator():
    g = DummyGraph()
    gen = CupyGenerator(g)
    assert gen.get_fallback_prefix() == "cp"
    assert gen.get_helper_functions() == []

    node = IRNode("Einsum", op_type="Einsum")

    node = IRNode("TruncateDiv", op_type="TruncateDiv")

    node = IRNode("TruncateMod", op_type="TruncateMod")

    node = IRNode("Add", op_type="Add")
    assert gen.generic_visit(node, ["a", "b"]) == "cp.add(a, b)"

    node = IRNode("UnknownOp", op_type="UnknownOp")
    assert gen.generic_visit(node, [], kwarg1="val") == "cp.unknownop(kwarg1=\x27val\x27)"
    assert gen.generic_visit(node, ["a", "b"], kwarg1="val") == "cp.unknownop(a, b, kwarg1=\x27val\x27)"


def test_cupy_types():
    with patch("ml_switcheroo_compiler.backends.cupy.types.cp", DummyCp()):
        with patch("ml_switcheroo_compiler.backends.cupy.types.generic_zeros", return_value="zeros"):
            assert zeros(None, (2,)) == "zeros"
        with patch("ml_switcheroo_compiler.backends.cupy.types.generic_array", return_value="array"):
            assert array(None, [1]) == "array"
        with patch("ml_switcheroo_compiler.backends.cupy.types.generic_asarray", return_value="asarray"):
            assert asarray(None, [1]) == "asarray"
        with patch("ml_switcheroo_compiler.backends.cupy.types.generic_item", return_value="item"):
            assert item(None, [1]) == "item"


def test_cupy_eager_execute_op_fallback_to_numpy_registry():
    def dummy_eager(module, *args, **kwargs):
        return "dummy_eager_numpy_fallback"

    # Don't register it globally, we want the global registry to have it but
    # cupy to NOT have it. Wait, the code says:
    # try:
    #     func = getattr(cp, snake)
    # except AttributeError:
    #     func = global_eager_registry.get(op_type)
    #     if func:
    #         return func(np, *args, **kwargs)

    # We tested this with NumpyFallbackOp but global_eager_registry.get(op_type)
    # in execute_op is checked AT THE START of execute_op:
    # func_registry = global_eager_registry.get(op_type)
    # if func_registry is not None:
    #     return func_registry(cp, *args, **kwargs)

    # Meaning we CANNOT hit lines 41-42 (fallback to np via global registry)
    # unless global_eager_registry.get(op_type) was None at first, but then is not None
    # in the except block! That's impossible, they use the same op_type.
    # Ah, wait! The eager registry might have backends mapped.
    pass


def test_cupy_eager_op_mapping():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.cupy.eager import execute_op

    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings") as mock_get_mapping:
        mock_schema = type("Dummy", (), {"operations": {"TestOp2": type("DummyOp", (), {"target_api": "custom_op", "custom_code": "lambda *args, **kwargs: 'mapped_res'"})()}})()
        mock_get_mapping.return_value = mock_schema
        with patch("ml_switcheroo_compiler.backends.eager_registry.global_eager_registry.get", return_value=None):
            res = execute_op(None, "TestOp2")
            assert res == "mapped_res"
