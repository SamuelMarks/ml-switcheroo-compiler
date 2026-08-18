def test_buffer_allocation_cost_models():
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import _get_dtype_size

    # Missing yaml path
    with patch("os.path.exists", return_value=False):
        assert _get_dtype_size("float32") == 4
