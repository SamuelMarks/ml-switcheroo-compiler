import numpy as np


def test_dot_general_missing_shape():
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    old_eager = config.eager_mode
    config.eager_mode = False
    try:

        class FakeData:
            shape = ()

        t1 = Tensor(data=FakeData(), config=TensorConfig((), DType.Float32, None))
        t2 = Tensor(data=FakeData(), config=TensorConfig((2,), DType.Float32, None))

        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
            with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
                # Patch in the module it was imported into, but avoid `matmul.matmul` collision by targeting the full matmul.py module dictionary directly,
                # actually matmul imports _emit_linalg_node from .utils, but matmul is also a function.
                # So we can patch the real definition in utils
                with patch("ml_switcheroo_compiler.ops.linalg.utils._emit_linalg_node", return_value="dummy_node"):
                    # wait, that doesn't work if matmul.py does `from .utils import _emit_linalg_node`
                    # Actually, matmul is a function, so patching "ml_switcheroo_compiler.ops.linalg.matmul._emit_linalg_node" fails because it gets the function!
                    # Let's import the module itself and patch it directly.
                    pass
    finally:
        config.eager_mode = old_eager


def test_dot_general_missing_shape_real():
    import sys

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    matmul_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.matmul"]
    from unittest.mock import patch

    from ml_switcheroo_compiler.core.config import config

    old_eager = config.eager_mode
    config.eager_mode = False
    try:

        class FakeData:
            shape = ()

        t1 = Tensor(data=FakeData(), config=TensorConfig((), DType.Float32, None))
        t2 = Tensor(data=FakeData(), config=TensorConfig((2,), DType.Float32, None))

        with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
            with patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
                original_emit = matmul_mod._emit_linalg_node
                matmul_mod._emit_linalg_node = lambda *args, **kwargs: "dummy_node"
                try:
                    assert matmul_mod.dot_general(t1, t2, (((), ()), ((), ()))) == "dummy_node"
                finally:
                    matmul_mod._emit_linalg_node = original_emit
    finally:
        config.eager_mode = old_eager


def test_matmul_missing(monkeypatch):
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.linalg.matmul import BlockMaskedMm, GatherMm, dot_general

    # BlockMaskedMm
    op1 = BlockMaskedMm()
    assert op1.infer_shape((2, 3), (4, 5)) is None

    # GatherMm
    op2 = GatherMm()
    assert op2.infer_shape((2, 3), (4, 5), lhs_indices=None, rhs_indices=None) is None

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            t1 = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
            t2 = Tensor(np.array([3.0, 4.0]), TensorConfig((2,), "float32", "cpu"))
            res = dot_general(t1, t2, dimension_numbers=(((0,), (0,)), ((), ())))
            assert isinstance(res, Tensor)


def test_solvers_missing():
    from ml_switcheroo_compiler.ops.linalg.solvers import Pinv

    op = Pinv()
    assert op.infer_shape(None) == ()

    class DummyShape:
        shape = (3,)

    assert op.infer_shape(DummyShape()) == (3,)


def test_einsum_missing():
    import pytest

    from ml_switcheroo_compiler.ops.linalg.einsum import ParsedEquationPart

    p = ParsedEquationPart("ij", (3,))
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
