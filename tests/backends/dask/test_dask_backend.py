from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.backends.dask.eager import execute_op
from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.dask.types import array, asarray, item, zeros
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRNode


class DummyGraph:
    def __init__(self):
        self.nodes = []


class DummyDa:
    def add(self, *args, **kwargs):
        return "add_res"


def test_dask_eager_execute_op():
    cp_mock = DummyDa()

    # Test registered func
    def dummy_eager(module, *args, **kwargs):
        return "dummy_eager_res"

    global_eager_registry.register("TestOpDask")(dummy_eager)

    res = execute_op(None, "TestOpDask")
    assert res == "dummy_eager_res"

    if "Add" in global_eager_registry._registry:
        del global_eager_registry._registry["Add"]

    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings") as mock_mappings:
        mock_schema = type("Dummy", (), {"operations": {"Add": type("DummyOp", (), {"target_api": "add", "custom_code": None})()}})()
        mock_mappings.return_value = mock_schema
        with patch("sys.modules", {"ml_switcheroo_compiler.backends.dask.eager": cp_mock}):
            res2 = execute_op(None, "Add")
            assert res2 == "add_res"

    if "UnknownOpDask" in global_eager_registry._registry:
        del global_eager_registry._registry["UnknownOpDask"]

    try:
        execute_op(None, "UnknownOpDask")
    except BackendNotSupportedError:
        pass


def test_dask_eager_execute_op_exception():
    if "OpThatRaisesDask" in global_eager_registry._registry:
        del global_eager_registry._registry["OpThatRaisesDask"]

    with pytest.raises(BackendNotSupportedError):
        execute_op(None, "OpThatRaisesDask")

    with patch("ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings") as mock_mappings:
        mock_schema = type("Dummy", (), {"operations": {"SnakeOp": type("DummyOp", (), {"target_api": "snake_op", "custom_code": None})()}})()
        mock_mappings.return_value = mock_schema
        with patch("sys.modules", {"ml_switcheroo_compiler.backends.dask.eager": DummyDa()}):
            with pytest.raises(BackendNotSupportedError):
                execute_op(None, "SnakeOp")


def test_dask_generator():
    g = DummyGraph()
    gen = DaskGenerator(g)
    assert gen.get_fallback_prefix() == "da"
    assert gen.get_helper_functions() == []

    node = IRNode("Einsum", op_type="Einsum")
    node = IRNode("TruncateDiv", op_type="TruncateDiv")
    node = IRNode("TruncateMod", op_type="TruncateMod")

    node = IRNode("Add", op_type="Add")
    assert gen.generic_visit(node, ["a", "b"]) == "da.add(a, b)"

    node = IRNode("UnknownOp", op_type="UnknownOp")
    assert gen.generic_visit(node, [], kwarg1="val") == "da.unknownop(kwarg1='val')"
    assert gen.generic_visit(node, ["a", "b"], kwarg1="val") == "da.unknownop(a, b, kwarg1='val')"


def test_dask_types():
    with patch("ml_switcheroo_compiler.backends.dask.types.da") as mock_da:
        mock_da.zeros.return_value = "zeros"
        assert zeros(None, (2,)) == "zeros"
        mock_da.array.return_value = "array"
        assert array(None, [1]) == "array"
        assert array(None, [1], dtype="float32") == "array"
        from unittest.mock import MagicMock

        mock_asarray_ret = MagicMock()
        mock_asarray_ret.compute.return_value.item.return_value = 1.0
        mock_da.asarray.return_value = mock_asarray_ret
        assert asarray(None, [1]) == mock_asarray_ret
        assert item(None, [1]) == 1.0
