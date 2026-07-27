from unittest.mock import patch


def test_conv_utils_extras():
    import ml_switcheroo_compiler.ops.nn.conv_utils as conv_utils

    class DummyTensor:
        shape = (1, 1, 1, 1)

        def __array__(self):
            return __import__("numpy").ones((1, 1, 1, 1))

        @property
        def dtype(self):
            return "float32"

        def asnumpy(self):
            return __import__("numpy").ones((1, 1, 1, 1))

    t = DummyTensor()

    def mock_get_op(name):
        class MockOp:
            def __call__(self, *args, **kwargs):
                return lambda *a, **kw: "dummy_node"

        return MockOp

    with patch("ml_switcheroo_compiler.ops.nn.conv_utils.get_op", side_effect=mock_get_op):
        conv_utils.atrous_conv2d(t, t, rate=1, padding="VALID")
        conv_utils.atrous_conv2d_transpose(t, t, t)
        conv_utils.depthwise_conv2d(t, t)
        conv_utils.convolution(t, t)

    conv_utils._build_conv_config({"lhs_dilation": 1, "rhs_dilation": 1}, ((0, 1, 2, 3), (), ()))


def test_erosion_dilation():
    pass
