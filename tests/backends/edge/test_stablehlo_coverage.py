"""Test StableHLO backend coverage."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator


def test_stablehlo_coverage():
    """Test stablehlo code generation edge cases."""
    g = LogicalGraph(outputs=["non_existent_output"])

    # 1. Test Input node evaluates to arg_name and is cached in var_map
    n1 = LogicalNode(id="n1", op_type="Input", shape_metadata=(1, 2))
    g.nodes["n1"] = n1

    # 2. Test missing input node for generic op (lines 109)
    # n2 has input "missing_input" which isn't in graph
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["missing_input"])
    g.nodes["n2"] = n2

    gen = StableHLOCodeGenerator(g)

    # Check generic_visit for Input
    arg_name = gen.generic_visit(n1, [])
    assert arg_name == "%arg0"

    # Check generic_visit for missing input type inference
    res_var = gen.generic_visit(n2, ["%missing_input"])
    assert res_var == "%v_n2"

    # Check generate module (lines 148, output_ids missing node)
    code = gen.generate()
    assert "tensor<f32>" in code  # from the missing output type fallback
