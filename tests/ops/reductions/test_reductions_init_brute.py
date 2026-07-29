def test_reductions_init_missing_imports():
    import importlib
    import sys
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.reductions

    ops_base_mod = sys.modules["ml_switcheroo_compiler.ops.base"]

    # We patch ml_switcheroo_compiler.ops.base.get_op safely
    with patch.object(ops_base_mod, "get_op", side_effect=KeyError("Boom")):
        # We need to forcefully reload ml_switcheroo_compiler.ops.reductions to hit KeyErrors
        importlib.reload(ml_switcheroo_compiler.ops.reductions)

        assert ml_switcheroo_compiler.ops.reductions.sum is None
        assert ml_switcheroo_compiler.ops.reductions.prod is None

    # After test, we should reload cleanly
    importlib.reload(ml_switcheroo_compiler.ops.reductions)
