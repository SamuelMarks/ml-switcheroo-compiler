def test_extract_proxy_inputs_elif_branch():
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTracingState:
        is_tracing = True
        active_graph = True

        def add_node(self, node):
            pass

    old_state = bmod.global_tracing_state
    bmod.global_tracing_state = FakeTracingState()

    class NonTensorWithId:
        def __init__(self):
            self.id = "my_proxy_id"
            self.shape = (4, 4)

    ids, shapes, first = TracingNodeBuilder.extract_proxy_inputs((NonTensorWithId(),))
    assert ids == ["my_proxy_id"]
    assert shapes == [(4, 4)]
    bmod.global_tracing_state = old_state
