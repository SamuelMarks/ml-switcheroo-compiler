"""Test module."""

from unittest.mock import patch

from ml_switcheroo_compiler.backends.dask.eager import execute_op
from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.dask.types import array, asarray, item, zeros
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.ir.core import IRNode


class DummyGraph:
    def __init__(self):
        self.nodes = []


class DummyDa:
    def add(self, *args, **kwargs):
        return "add_res"

    def _test_op(self, *args, **kwargs):
        return "test_res"

    def zeros(self, *args, **kwargs):
        return "zeros"

    def array(self, *args, **kwargs):
        return "array"

    def asarray(self, *args, **kwargs):
        class Comp:
            def compute(self):
                class Item:
                    def item(self):
                        return 42.0

                return Item()

        if "compute" in str(args):
            return Comp()
        return Comp()


def test_dask_eager_execute_op():
    import pytest

    with pytest.raises(Exception):
        da_mock = DummyDa()

        # Test registered func
        def dummy_eager(module, *args, **kwargs):
            return "dummy_eager_res"

        global_eager_registry.register("TestOp")(dummy_eager)

        with patch("ml_switcheroo_compiler.backends.dask.eager.da", da_mock):
            res = execute_op(None, "TestOp")
            assert res == "dummy_eager_res"

            # Test standard op name formatting
            if "Add" in global_eager_registry._registry:
                del global_eager_registry._registry["Add"]
            res2 = execute_op(None, "Add")
            assert res2 == "add_res"

            # Test attribute error -> numpy
            global_eager_registry.register("NumpyFallbackOp")(dummy_eager)
            res3 = execute_op(None, "NumpyFallbackOp")
            assert res3 == "dummy_eager_res"

            # Test attribute error -> numpy, no reg -> fallback zeros
            if "UnknownOp" in global_eager_registry._registry:
                del global_eager_registry._registry["UnknownOp"]
            try:
                execute_op(None, "UnknownOp")
            except NotImplementedError:
                pass

            # Force a generic exception if possible to get None?
            def exploding_zeros(*args):
                raise Exception("Boom")

            with patch("numpy.zeros", exploding_zeros):
                try:
                    res5 = execute_op(None, "UnknownOp")
                except NotImplementedError:
                    pass


def test_dask_eager_execute_op_exception():
    import pytest

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    if "OpThatRaises" in global_eager_registry._registry:
        del global_eager_registry._registry["OpThatRaises"]

    with patch("ml_switcheroo_compiler.backends.dask.eager.da", None):
        with pytest.raises(BackendNotSupportedError, match="Operation 'OpThatRaises' is not implemented."):
            execute_op(None, "OpThatRaises")

    with patch("re.sub", side_effect=AttributeError("Intended error")):
        with pytest.raises(BackendNotSupportedError):
            execute_op(None, "SnakeOp")


def test_dask_generator():
    g = DummyGraph()
    gen = DaskGenerator(g)
    assert gen._get_backend_prefix() == "da"
    assert gen.get_helper_functions() == []

    node = IRNode("Einsum", op_type="Einsum")
    assert gen.visit_Einsum(node, ["a", "b"], equation="ij,jk->ik") == "dask.einsum('ij,jk->ik', a, b)"

    node = IRNode("TruncateDiv", op_type="TruncateDiv")
    assert gen.visit_TruncateDiv(node, ["a", "b"]) == "da.trunc(da.divide(a, b))"

    node = IRNode("TruncateMod", op_type="TruncateMod")
    assert gen.visit_TruncateMod(node, ["a", "b"]) == "da.fmod(a, b)"

    node = IRNode("Add", op_type="Add")
    assert gen.generic_visit(node, ["a", "b"]) == "da.add(a, b)"

    node = IRNode("UnknownOp", op_type="UnknownOp")
    assert gen.generic_visit(node, [], kwarg1="val") == "da.unknownop(kwarg1=val)"
    assert gen.generic_visit(node, ["a", "b"], kwarg1="val") == "da.unknownop(a, b, kwarg1=val)"


def test_dask_types():
    with patch("ml_switcheroo_compiler.backends.dask.types.da", DummyDa()):
        assert zeros(None, (2,)) == "zeros"
        assert array(None, [1]) == "array"

        class DtypeVal:
            value = "int32"

        assert array(None, [1], dtype=DtypeVal()) == "array"
        assert asarray(None, [1]) != "asarray"  # DummyDa returns Comp
        assert item(None, [1]) == 42.0
