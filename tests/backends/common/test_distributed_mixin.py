"""Tests for test_distributed_mixin."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from ml_switcheroo_ir import AttributeValue, IRGraph

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.mixins.distributed import DistributedASTVisitor
from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode
from tests.testing_utils import MockNodeStub

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class DummyGenerator(BaseGenerator):
    """Test class DummyGenerator."""

    def __init__(self, name: BackendArg) -> None:
        """Test function __init__.

        Args:
            name: Test parameter `name`.
        """
        super().__init__(IRGraph())
        type(self).__name__ = str(name)
        self.code: list[str] = []

    def generate(self, graph: IRGraph | None = None) -> str:
        """Emit empty generated source.

        Args:
            graph: Optional IR graph.

        Returns:
            str: Empty string.
        """
        del graph
        return ""


class DummyNode(MockNodeStub):
    """Test class DummyNode."""

    def __init__(
        self,
        attrs: Mapping[str, AttributeValue] | None = None,
        shape: tuple[int | str | SymNode | SymInt | None, ...] | None = None,
        dtype: str | None = "float32",
    ) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.
        """
        from ml_switcheroo_ir import DType

        from ml_switcheroo_compiler.ir.core import normalize_shape_metadata

        self.attributes = dict(attrs) if attrs is not None else {}
        self.shape_metadata = normalize_shape_metadata(shape) or (1,)
        self.dtype = DType.from_str(dtype) if dtype is not None else None


def test_distributed_ast_visitor() -> None:
    """Test function test_distributed_ast_visitor."""
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
