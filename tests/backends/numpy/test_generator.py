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


def test_numpy_generator_save_load(tmp_path):
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = NumpyGenerator(IRGraph())
    arr = np.array([1, 2, 3])

    # save/load
    file1 = tmp_path / "test.npy"
    gen.save(file1, arr)
    res1 = gen.load(file1)
    np.testing.assert_array_equal(res1, arr)

    # savez
    file2 = tmp_path / "test2.npz"
    gen.savez(file2, a=arr)
    res2 = gen.load(file2)
    np.testing.assert_array_equal(res2["a"], arr)

    # savez_compressed
    file3 = tmp_path / "test3.npz"
    gen.savez_compressed(file3, a=arr)
    res3 = gen.load(file3)
    np.testing.assert_array_equal(res3["a"], arr)


def test_numpy_generator_get_rng():
    from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    rng = NumpyGenerator.get_numpy_rng(42)
    assert rng is not None

    gen = NumpyGenerator(IRGraph())
    assert gen._get_backend_prefix() == "np"
    assert gen.get_helper_functions() == []
