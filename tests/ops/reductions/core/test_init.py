def test_reductions_init():
    import ml_switcheroo_compiler.ops.reductions as red

    assert red.sum is not None


def test_reductions_init_missing():
    import importlib
    import sys

    ops_base = sys.modules["ml_switcheroo_compiler.ops.base"]
    old_get_op = ops_base.get_op

    def mock_get_op(*args, **kwargs):
        raise KeyError("mock")

    ops_base.get_op = mock_get_op

    if "ml_switcheroo_compiler.ops.reductions" in sys.modules:
        del sys.modules["ml_switcheroo_compiler.ops.reductions"]
    import ml_switcheroo_compiler.ops.reductions

    try:
        assert getattr(ml_switcheroo_compiler.ops.reductions, "sum", None) is None
    finally:
        ops_base.get_op = old_get_op
        importlib.reload(ml_switcheroo_compiler.ops.reductions)
