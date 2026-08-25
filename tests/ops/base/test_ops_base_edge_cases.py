from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.ops.base import emit_ir_node


def test_emit_ir_node():
    # with graph
    graph = MagicMock()
    graph.nodes = {}
    nid = emit_ir_node(graph, "TestOp", ["in1"], shape_metadata=(1,))
    assert nid in graph.nodes

    # without graph
    with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state") as global_state:
        nid = emit_ir_node(None, "TestOp", ["in1"])
        global_state.add_node.assert_called_once()


from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import OpDef, dispatch_eager


def test_opdef_base_methods():
    op = OpDef()
    assert op.infer_shape() == tuple()

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
        backend = MagicMock()
        backend.execute_op.return_value = "eager_result"
        mock_get_backend.return_value = backend
        assert op.eager_eval() == "eager_result"


def test_dispatch_eager_wrapper():
    @dispatch_eager("TestOp")
    def test_func(t1, t2):
        return "not eager"

    with patch("ml_switcheroo_compiler.ops.base.config") as mock_config:
        # Test not eager mode
        mock_config.eager_mode = False
        assert test_func(1, 2) == "not eager"

        # Test eager mode
        mock_config.eager_mode = True
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_get_backend:
            backend = MagicMock()
            backend.array.return_value = MagicMock(shape=(1,))
            backend.execute_op.return_value = "backend_res"
            mock_get_backend.return_value = backend

            # with tensor inputs
            t1 = MagicMock(spec=Tensor)
            t1.data = "t1_data"
            t1.device = "dev"
            from ml_switcheroo_compiler.core.dtype import DType

            t1.dtype = DType.Float32
            res = test_func(t1, 2)
            assert isinstance(res, Tensor)

            # with multiple tensor return
            backend.execute_op.return_value = ("backend_res1", "backend_res2")
            res_tuple = test_func(t1, 2)
            assert isinstance(res_tuple, tuple)
            assert isinstance(res_tuple[0], Tensor)
            assert isinstance(res_tuple[1], Tensor)
