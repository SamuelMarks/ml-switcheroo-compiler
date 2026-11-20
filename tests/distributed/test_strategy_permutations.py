"""Tests for test_strategy_permutations."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import NodeDict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_strategy_exhaustive_extra() -> None:
    """Test function test_strategy_exhaustive_extra."""
    from unittest.mock import mock_open, patch

    import yaml
    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.distributed.strategy import (
        MultiWorkerMirroredStrategy,
        ParameterServerStrategy,
        _load_strategy_config,
        _load_webrtc_topology,
    )

    # Test load yaml
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump({"strategies": {"test": "val"}}))):
            res = _load_strategy_config()
            assert res == {"test": "val"}

        with patch("builtins.open", mock_open(read_data=yaml.dump({"test2": "val2"}))):
            res2 = _load_webrtc_topology()
            assert res2 == {"test2": "val2"}

    # Test PS Hooks
    ps = ParameterServerStrategy()

    class MockBackend:
        """Test class MockBackend."""

        def ps_pull_weights(self, graph: BackendArg, resolver: BackendArg) -> BackendArg:
            """Test function ps_pull_weights.

            Args:
                graph: Test parameter `graph`.
                resolver: Test parameter `resolver`.

            Returns:
                BackendArg: Computed output value from `ps_pull_weights`.
            """
            del graph, resolver
            return True

        def ps_push_gradients(self, graph: BackendArg, resolver: BackendArg) -> BackendArg:
            """Test function ps_push_gradients.

            Args:
                graph: Test parameter `graph`.
                resolver: Test parameter `resolver`.

            Returns:
                BackendArg: Computed output value from `ps_push_gradients`.
            """
            del graph, resolver
            return True

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackend()):
        assert ps.pull_weights(IRGraph()) is True
        assert ps.push_gradients(IRGraph()) is True

    # Test PS rewire consumers
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Constant")
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])
    g.nodes = NodeDict({"n1": n1, "n2": n2})

    ps = ParameterServerStrategy()
    ps.pull_weights(g)
    assert "n1_recv" in g.nodes
    assert g.nodes["n2"].inputs == ["n1_recv"]

    # MWMS Hooks
    mwms = MultiWorkerMirroredStrategy()

    class MockBackendMWMS:
        """Test class MockBackendMWMS."""

        def sync_gradients_hook(self, graph: BackendArg, resolver: BackendArg) -> BackendArg:
            """Test function sync_gradients_hook.

            Args:
                graph: Test parameter `graph`.
                resolver: Test parameter `resolver`.

            Returns:
                BackendArg: Computed output value from `sync_gradients_hook`.
            """
            del graph, resolver
            return True

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackendMWMS()):
        assert mwms.sync_gradients(IRGraph()) is True
