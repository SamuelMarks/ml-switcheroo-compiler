"""Coverage tests for custom rules autodiff."""

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import (
    _assoc_scan_vjp,
    _if_vjp,
    _loop_vjp,
    _scan_vjp,
)


def test_if_vjp():
    assert _if_vjp(None, None, None) == (UnconnectedGradients.ZERO,)


def test_loop_vjp():
    node = LogicalNode(id="n1", op_type="Loop", inputs=["a", "b"])
    assert _loop_vjp(None, node, None) == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)


def test_scan_vjp():
    node = LogicalNode(id="n1", op_type="Scan", inputs=["a"])
    assert _scan_vjp(None, node, None) == (UnconnectedGradients.ZERO,)


def test_assoc_scan_vjp():
    node = LogicalNode(id="n1", op_type="AssociativeScan", inputs=["a", "b", "c"])
    assert _assoc_scan_vjp(None, node, None) == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)


def test_jvp_nulls():
    pass
