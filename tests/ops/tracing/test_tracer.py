from ml_switcheroo_compiler.tracing.tracer import TracerTape, get_trace_count, increment_trace_count, reset_trace_count


def test_tracer_coverage():
    def mock_func():
        pass

    reset_trace_count(mock_func)
    assert get_trace_count(mock_func) == 0
    increment_trace_count(mock_func)
    assert get_trace_count(mock_func) == 1
    reset_trace_count(mock_func)
    reset_trace_count(mock_func)
    assert get_trace_count(mock_func) == 0
    reset_trace_count(mock_func)

    tape = TracerTape()
    tape.start_tracing()
    tape.add_node(type("obj", (object,), {"id": "1"})())
    tape.stop_tracing()
