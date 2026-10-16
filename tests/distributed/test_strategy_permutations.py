"""Tests for test_strategy_permutations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_strategy_exhaustive_extra() -> None:
    """Test function test_strategy_exhaustive_extra."""
    from unittest.mock import mock_open, patch

    import yaml

    from ml_switcheroo_compiler.distributed.strategy import (
        MultiWorkerMirroredStrategy,
        ParameterServerStrategy,
        _load_strategy_config,
        _load_webrtc_topology,
    )
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    # Test load yaml
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump({"strategies": {"test": "val"}}))):
            res = _load_strategy_config()
            assert cast("EagerTensorProtocol", res) == cast("EagerTensorProtocol", {"test": "val"})

        with patch("builtins.open", mock_open(read_data=yaml.dump({"test2": "val2"}))):
            res2 = cast("EagerTensorProtocol", _load_webrtc_topology())
            assert res2 == cast("EagerTensorProtocol", {"test2": "val2"})

    # Test PS Hooks
    ps = ParameterServerStrategy()
    ps.config = {"registry_hooks": {"pull": "mock_pull", "push": "mock_push"}}

    class MockBackend:
        """Test class MockBackend."""

        def mock_pull(self, graph: EagerValue, resolver: EagerValue) -> EagerValue:
            """Test function mock_pull."""
            return True

        def mock_push(self, graph: EagerValue, resolver: EagerValue) -> EagerValue:
            """Test function mock_push."""
            return True

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackend()):
        assert ps.pull_weights(IRGraph()) is True
        assert ps.push_gradients(IRGraph()) is True

    # Test PS rewire consumers
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Constant")
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])
    g.nodes = {"n1": n1, "n2": n2}

    ps = ParameterServerStrategy()
    ps.pull_weights(g)
    assert "n1_recv" in g.nodes
    assert g.nodes["n2"].inputs == ["n1_recv"]

    # MWMS Hooks
    mwms = MultiWorkerMirroredStrategy()
    mwms.config = {"registry_hooks": {"sync": "mock_sync"}}

    class MockBackendMWMS:
        """Test class MockBackendMWMS."""

        def mock_sync(self, graph: EagerValue, resolver: EagerValue) -> EagerValue:
            """Test function mock_sync."""
            return True

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackendMWMS()):
        assert mwms.sync_gradients(IRGraph()) is True
