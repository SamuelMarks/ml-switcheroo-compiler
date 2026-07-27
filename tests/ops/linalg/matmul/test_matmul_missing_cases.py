def test_matmul_missing(monkeypatch):
    import ml_switcheroo_compiler.ops.linalg.utils as linalg_utils
    import ml_switcheroo_compiler.tracing.builder as builder
    from ml_switcheroo_compiler.ops.linalg.matmul import BlockMaskedMm, GatherMm, dot_general

    # BlockMaskedMm
    op1 = BlockMaskedMm()
    assert op1.infer_shape((2, 3), (4, 5)) is None

    # GatherMm
    op2 = GatherMm()
    assert op2.infer_shape((2, 3), (4, 5), lhs_indices=None, rhs_indices=None) is None

    # DotGeneral when shapes are missing
    class Dummy:
        shape = None
        dtype = "float32"
        device = "cpu"

        def __init__(self):
            pass

    class FakeTracingState:
        is_tracing = True
        active_graph = True

        def add_node(self, node):
            pass

    monkeypatch.setattr(linalg_utils, "global_tracing_state", FakeTracingState())
    monkeypatch.setattr(builder, "global_tracing_state", FakeTracingState())

    with monkeypatch.context() as m:
        m.setattr(linalg_utils.TracingNodeBuilder, "extract_proxy_inputs", lambda x: (["id1", "id2"], [], []))
        import uuid

        m.setattr(uuid, "uuid4", lambda: "mock_uuid")

        node = dot_general(Dummy(), Dummy(), dimension_numbers=(((0,), (0,)), ((), ())))


def test_solvers_missing():
    from ml_switcheroo_compiler.ops.linalg.solvers import Pinv

    op = Pinv()
    assert op.infer_shape(None) == ()

    class DummyShape:
        shape = (3,)

    assert op.infer_shape(DummyShape()) == (3,)


def test_einsum_missing():
    from ml_switcheroo_compiler.ops.linalg.einsum import ParsedEquationPart

    p = ParsedEquationPart("ij", (3,))
    import pytest

    with pytest.raises(ValueError):
        p._check_dimension_mismatch({"i": 2}, "i", 3)


def test_einsum_missing_branch():
    from ml_switcheroo_compiler.ops.linalg.einsum import ParsedEquationPart

    p = ParsedEquationPart("ij", (3, 1))
    p.shape = (3, 1)
    p.chars = ["i", "i"]
    axis_map = {"i": 3}
    p.process_axis_map(axis_map)
    assert axis_map["i"] == 3
