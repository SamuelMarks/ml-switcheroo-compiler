def test_reductions_init():
    import ml_switcheroo_compiler.ops.reductions as red

    assert red.sum is not None


def test_reductions_init_missing():
    import importlib
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.reductions

    with patch("ml_switcheroo_compiler.ops.base.get_op", side_effect=KeyError("mock")):
        importlib.reload(ml_switcheroo_compiler.ops.reductions)
        assert getattr(ml_switcheroo_compiler.ops.reductions, "sum", None) is None

    importlib.reload(ml_switcheroo_compiler.ops.reductions)
