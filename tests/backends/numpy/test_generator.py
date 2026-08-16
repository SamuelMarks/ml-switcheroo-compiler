"""Test Numpy generator edge cases coverage."""

from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor


def test_numpy_generator_triinv():
    """Test visit_TriInv string generation."""
    # We can just call it on the class directly since it's a @classmethod
    # TriInv does not output kwargs in its logic, but we can verify it doesn't break
    res = NumpyASTVisitor.visit_TriInv(None, ["in_val"], dimension=0, equation="test")
    assert res == "np.linalg.inv(in_val)"


def test_numpy_generator_format_kwargs_dimension():
    """Test _format_kwargs with dimension keyword arg."""
    res = NumpyASTVisitor._format_kwargs({"dimension": 2, "other": "test"})
    assert "axis=2" in res
    assert "other=test" in res
    assert "dimension" not in res


def test_numpy_generator_generic_visit_dimension():
    from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor
    from ml_switcheroo_compiler.ir.core import LogicalNode

    node = LogicalNode(id="n1", op_type="Sum", inputs=["x"])
    res = NumpyASTVisitor.generic_visit(node, ["x"], dimension=1)
    assert res == "np.sum(x, axis=1)"


def test_numpy_generator_get_rng():
    from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator

    rng = NumpyGenerator.get_numpy_rng(42)
    assert rng is not None
