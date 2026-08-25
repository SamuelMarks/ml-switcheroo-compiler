from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor, NumpyGenerator, NumpyTypeTranslator


def test_numpy_type_translator():
    assert NumpyTypeTranslator.get_fallback_prefix() == "np"


def test_numpy_ast_visitor():
    kwargs = {"equation": "ij,jk->ik", "dimension": 1, "other": "val"}
    assert NumpyASTVisitor._format_kwargs(kwargs) == "other=val, axis=1"

    node = MagicMock(op_type="Parameter")
    node.id = "p1"
    assert NumpyASTVisitor.visit_Parameter(node, ["p1"]) == "p1 = None # Parameter"

    assert NumpyASTVisitor.visit_Return(None, []) == "return None"
    assert NumpyASTVisitor.visit_Return(None, ["v1"]) == "return v1"
    assert NumpyASTVisitor.visit_Return(None, ["v1", "v2"]) == "return v1, v2"

    assert NumpyASTVisitor.visit_TriInv(None, ["a"]) == "np.linalg.inv(a)"
    assert NumpyASTVisitor.visit_TruncateDiv(None, ["a", "b"]) == "np.trunc(np.divide(a, b))"
    assert NumpyASTVisitor.visit_TruncateMod(None, ["a", "b"]) == "np.fmod(a, b)"

    node = MagicMock(op_type="Sum")
    assert NumpyASTVisitor.generic_visit(node, ["a"], dimension=1) == "np.sum(a, axis=1)"
    assert NumpyASTVisitor.generic_visit(node, []) == "np.sum()"
    assert NumpyASTVisitor.generic_visit(node, ["a"], keepdims=True) == "np.sum(a, keepdims=True)"


def test_numpy_generator():
    gen = NumpyGenerator(MagicMock(nodes=[]))
    assert gen.get_fallback_prefix() == "np"
    assert gen._get_backend_prefix() == "np"

    node = MagicMock(op_type="PowerIteration")
    node.attributes = {"num_iters": 5}
    assert gen.visit_PowerIteration(node, ["a"]) == "np_power_iteration(a, 5, None)"
    assert gen.visit_PowerIteration(node, ["a", "b"]) == "np_power_iteration(a, 5, b)"

    assert gen.get_numpy_rng() is not None

    with patch("numpy.load", return_value="load"):
        assert gen.load("file.npy") == "load"

    with patch("numpy.save"):
        gen.save("file.npy", "arr")

    with patch("numpy.savez"):
        gen.savez("file.npz", "arr")

    with patch("numpy.savez_compressed"):
        gen.savez_compressed("file.npz", "arr")


def test_numpy_generator_helpers():
    gen = NumpyGenerator(MagicMock(nodes=[]))
    assert isinstance(gen.get_helper_functions(), list)
