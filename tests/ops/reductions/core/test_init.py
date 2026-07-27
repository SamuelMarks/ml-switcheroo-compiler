import importlib
from unittest.mock import patch


def test_reductions_init():
    import ml_switcheroo_compiler.ops.reductions as red

    importlib.reload(red)

    assert red.sum is not None


def test_reductions_init_keyerror():
    # To test the KeyError branches in ml_switcheroo_compiler.ops.reductions.__init__,
    # we can mock ml_switcheroo_compiler.ops.registry.get_op
    # and then manually execute the code from __init__.py.
    # Alternatively, just patch it and reload.

    def mock_get_op(name):
        raise KeyError(name)

    with patch("ml_switcheroo_compiler.ops.registry.get_op", mock_get_op):
        # because the module imports `get_op` from `ml_switcheroo_compiler.ops.base`,
        # which imports it from `ml_switcheroo_compiler.ops.registry`,
        # but `ops.base` might have already imported it.
        # So let's patch `ml_switcheroo_compiler.ops.base.get_op` directly in the module dict of base:
        # wait, as we saw, ops.base is shadowed by unary.base.
        # Let's import the actual module file.
        import sys

        actual_base = sys.modules["ml_switcheroo_compiler.ops.base"]

        with patch.object(actual_base, "get_op", mock_get_op):
            import ml_switcheroo_compiler.ops.reductions as red

            importlib.reload(red)

            assert red.sum is None
            assert red.mean is None
            assert red.logsumexp is None
