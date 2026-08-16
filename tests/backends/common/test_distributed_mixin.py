from ml_switcheroo_compiler.backends.common.mixins.distributed import DistributedASTVisitor


class DummyGenerator:
    def __init__(self, name):
        self.__class__.__name__ = name
        self.code = []


class DummyNode:
    def __init__(self, attrs=None, shape=None, dtype="float32"):
        self.attributes = attrs or {}
        self.shape_metadata = shape or (1,)
        self.dtype = dtype


def test_distributed_ast_visitor():
    for name, expected_send, expected_recv in [
        ("NumpyGenerator", "_numpy_send(v0, target=1)", "_numpy_recv(source=1, shape=(1,), dtype='float32')"),
        ("PytorchCodeGenerator", "torch.distributed.isend(v0, dst=1)", "torch.distributed.irecv(src=1)"),
        ("TorchGenerator", "torch.distributed.isend(v0, dst=1)", "torch.distributed.irecv(src=1)"),
        ("JaxGenerator", "jax.lax.send(v0, dst=1)", "jax.lax.recv(src=1)"),
        ("MlxGenerator", "mlx.core.distributed.send(v0, dst=1)", "mlx.core.distributed.recv(src=1)"),
        ("KerasGenerator", "keras.distribution.send(v0, target=1)", "keras.distribution.recv(source=1)"),
        ("UnknownGenerator", "send(v0, target=1)", "recv(source=1)"),
    ]:
        gen = DummyGenerator(name)
        visitor = DistributedASTVisitor(gen)

        send_node = DummyNode({"target_stage": 1})
        res_send = visitor.visit_Send(send_node, ["v0"])
        assert res_send == expected_send
        assert len(gen.code) == 1
        assert "target" in gen.code[-1] or "stage" in gen.code[-1]

        recv_node = DummyNode({"source_stage": 1})
        res_recv = visitor.visit_Recv(recv_node, [])
        assert res_recv == expected_recv
        assert len(gen.code) == 2
        assert "source" in gen.code[-1] or "stage" in gen.code[-1]
