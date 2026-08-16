def test_generated_registry():
    import ml_switcheroo_compiler.ops.generated_registry as gr

    assert "OPS_REGISTRY" in dir(gr)
    assert len(gr.OPS_REGISTRY) > 0
