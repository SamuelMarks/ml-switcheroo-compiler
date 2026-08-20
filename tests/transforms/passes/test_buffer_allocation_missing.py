def test_buffer_allocation_symbolic_empty_dims():
    from unittest.mock import PropertyMock, patch

    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import _get_node_byte_size

    node = IRNode("test", "TestOp")
    node.shape_metadata = ()
    node.attributes["dtype"] = "float32"

    with patch.object(IRNode, "is_dynamic_shape", new_callable=PropertyMock) as mock_is_dynamic:
        mock_is_dynamic.return_value = True
        assert _get_node_byte_size(node) == "4"


def test_get_dtype_size_fallback():
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.passes.buffer_allocation import _get_dtype_size

    with patch("os.path.exists", return_value=False):
        assert _get_dtype_size("float32") == 4
