from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.random.continuous.dirichlet import dirichlet


def test_dirichlet(mocker):
    # Mock global_tracing_state.add_node inside random.state
    mocker.patch("ml_switcheroo_compiler.random.state.global_tracing_state.add_node")

    class MockTensor:
        def __init__(self):
            self.data = type("M", (), {"id": "1"})()
            self.shape = ()

    t = MockTensor()
    res = dirichlet(t, t)
    assert res is not None

    res2 = dirichlet(t, t, shape=(2, 3), dtype=DType.Float64)
    assert res2 is not None
