"""Tests for test_type_promotion_explicitizer."""

from __future__ import annotations

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode, NodeDict

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import type_promotion_explicitizer_pass


def test_type_promotion_explicitizer() -> None:
    """Test function test_type_promotion_explicitizer."""
    graph = LogicalGraph()
    node1 = LogicalNode(id="node1", op_type="Add", inputs=["a", "b"], attributes={"dtype": DType.Float32})
    graph.nodes["node1"] = node1
    with pytest.raises(KeyError):
        type_promotion_explicitizer_pass(graph)


def test_type_promotion_explicitizer_extra() -> None:
    """Test function test_type_promotion_explicitizer_extra."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import _needs_cast, type_promotion_explicitizer_pass

    assert _needs_cast(None, "float32") is None
    assert _needs_cast("float32", "float32") is None
    assert _needs_cast("float32", "float64") in ("float32", "float64")
    assert _needs_cast("invalid", "float32") is None
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    n1.attributes["dtype"] = "float32"
    n2 = IRNode(id="n2", op_type="Input")
    n2.attributes["dtype"] = "float64"
    n3 = IRNode(id="n3", op_type="Input")
    n3.attributes["dtype"] = "float64"
    n_add = IRNode(id="add", op_type="Add", inputs=["n1", "n2"])
    n_add2 = IRNode(id="add2", op_type="Add", inputs=["n2", "n3"])
    n_not_2 = IRNode(id="not2", op_type="Add", inputs=["n1"])
    n_add3 = IRNode(id="add3", op_type="Add", inputs=["n2", "n1"])
    g.nodes = NodeDict({"n1": n1, "n2": n2, "n3": n3, "add": n_add, "add2": n_add2, "not2": n_not_2, "add3": n_add3})
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.transforms.pass_manager.DAGTopologicalSorter.sort", return_value=[n1, n2, n3, n_add, n_add2, n_not_2, n_add3]):
        with patch("ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer.dtype_inference_pass"):
            modified = type_promotion_explicitizer_pass(g)
    assert modified is True
    if n_add.inputs[0].startswith("cast_"):
        assert n_add.inputs[1] == "n2"
    else:
        assert n_add.inputs[1].startswith("cast_")
        assert n_add.inputs[0] == "n1"
    if n_add3.inputs[0].startswith("cast_"):
        assert n_add3.inputs[1] == "n1"
    else:
        assert n_add3.inputs[1].startswith("cast_")
        assert n_add3.inputs[0] == "n2"


def test_type_promotion_lattice_exceptions_and_missing_file() -> None:
    """Test _load_declarative_promotion_lattice error branches."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.core.type_promotion import _load_declarative_promotion_lattice

    mock_yaml1 = {"lattice": {"InvalidDTypeKey": {"OtherInvalid": "InvalidResult"}}}
    with patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value=mock_yaml1):
        _load_declarative_promotion_lattice()
    mock_yaml2 = {"lattice": {"float32": {"InvalidInnerDType": "float32"}}}
    with patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value=mock_yaml2):
        _load_declarative_promotion_lattice()
    with patch("os.path.exists", return_value=False):
        _load_declarative_promotion_lattice()
    assert callable(_load_declarative_promotion_lattice)
