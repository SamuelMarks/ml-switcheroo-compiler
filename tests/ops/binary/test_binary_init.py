def test_binary_init_funcs():
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.binary as binary

    with patch("ml_switcheroo_compiler.ops.creation.zeros_like", return_value="zeros"):
        with patch("ml_switcheroo_compiler.ops.shape.indexing.where", return_value="where"):
            with patch("ml_switcheroo_compiler.ops.binary.divide", return_value="divide"):
                with patch("ml_switcheroo_compiler.ops.binary.equal", return_value="equal"):
                    res = binary.divide_no_nan(1, 2)
                    assert res == "where"

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = "res"
        assert binary.polar(1, 2) == "res"
        assert binary.view_as_complex(1) == "res"
        assert binary.view_as_real(1) == "res"
