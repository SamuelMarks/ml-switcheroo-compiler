import pytest

from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor, NumpyGenerator, NumpyTypeTranslator


def test_numpy_generator_coverage():
    class DummyNode:
        def __init__(self, op_type, id="test_id"):
            self.op_type = op_type
            self.id = id
            self.attributes = {}

    class DummyGraph:
        def __init__(self):
            self.nodes = []

    # test NumpyTypeTranslator
    assert NumpyTypeTranslator.get_fallback_prefix() == "np"
    assert isinstance(NumpyTypeTranslator.get_ops_map(), dict)

    # test NumpyASTVisitor
    assert NumpyASTVisitor._format_kwargs({"a": 1}) == "a=1"
    assert NumpyASTVisitor.visit_TriInv(None, ["x"]) == "np.linalg.inv(x)"
    assert NumpyASTVisitor.visit_TruncateDiv(None, ["x", "y"]) == "np.trunc(np.divide(x, y))"
    assert NumpyASTVisitor.visit_TruncateMod(None, ["x", "y"]) == "np.fmod(x, y)"
    assert NumpyASTVisitor.generic_visit(DummyNode("Unknown"), []) == "np.unknown()"

    # test NumpyGenerator
    gen = NumpyGenerator(DummyGraph())

    node = DummyNode("Einsum")
    assert gen.visit_Einsum(node, ["a", "b"], equation="i,j->ij") == "np.einsum('i,j->ij', a, b)"

    node = DummyNode("PowerIteration")
    assert gen.visit_PowerIteration(node, ["w"]) == "np_power_iteration(w, 1, None)"
    assert gen.visit_PowerIteration(node, ["w", "u"]) == "np_power_iteration(w, 1, u)"

    assert gen.get_fallback_prefix() == "np"

    ops = gen.get_ops_map({})
    assert isinstance(ops, dict)

    assert gen._get_backend_prefix() == "np"

    helpers = gen.get_helper_functions()
    assert isinstance(helpers, list)

    with pytest.raises(Exception):
        NumpyGenerator.load("")
    NumpyGenerator.save("dummy.npy", None)
    NumpyGenerator.savez("dummy.npz")
    NumpyGenerator.savez_compressed("dummy2.npz")
