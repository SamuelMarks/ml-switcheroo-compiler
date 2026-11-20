"""Behavioral tests for `rand`, `randn`, `randint`, and `manual_seed` in eager and IR tracing modes."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager.random_ops import seed_rng
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.api_random import manual_seed, rand, randint, randn
from ml_switcheroo_compiler.tracing.tracer import TracerTape
from tests.testing_utils import MockTestStub, ensure_tensor


def test_random_creation_eager_execution() -> None:
    """Verify `rand`, `randn`, `randint`, and `manual_seed` execute eagerly on the NumPy backend."""
    with ConfigContext(backend="numpy", eager_mode=True):
        assert manual_seed(1234) == 1234

        class _ManualSeedBackend(MockTestStub):
            """Backend stub exposing `manual_seed`."""

            last_seed: int = -1

            def manual_seed(self, val: int) -> None:
                """Record applied seed value.

                Args:
                    val (int): Applied seed.
                """
                self.last_seed = val

        class _NoRandomMethodSubmod:
            """Random submodule lacking `.seed`."""

        class _NoSeedBackend(MockTestStub):
            """Backend stub with `.random` lacking `.seed` and no `manual_seed`."""

            random = _NoRandomMethodSubmod()

        ms_be = _ManualSeedBackend()
        assert seed_rng(ms_be, 77) == 77
        assert ms_be.last_seed == 77
        assert seed_rng(_NoSeedBackend(), 88) == 88

        r_def = ensure_tensor(rand(2, 3))
        assert r_def.shape == (2, 3)

        r_f64 = ensure_tensor(rand(2, 3, dtype=DType.Float64, device=Device("cpu")))
        assert r_f64.shape == (2, 3)
        assert r_f64.dtype == DType.Float64

        rn_def = ensure_tensor(randn(2, 3))
        assert rn_def.shape == (2, 3)

        rn_f64 = ensure_tensor(randn(2, 3, dtype=DType.Float64, device=Device("cpu")))
        assert rn_f64.shape == (2, 3)
        assert rn_f64.dtype == DType.Float64

        ri_def = ensure_tensor(randint(0, 10, (2, 3)))
        assert ri_def.shape == (2, 3)

        ri_i32 = ensure_tensor(randint(0, 10, [2, 3], dtype=DType.Int32, device=Device("cpu")))
        assert ri_i32.shape == (2, 3)
        assert ri_i32.dtype == DType.Int32


def test_random_creation_symbolic_tracing_emission() -> None:
    """Verify `rand`, `randn`, `randint`, and `manual_seed` emit canonical IR nodes during tracing."""
    with ConfigContext(eager_mode=False):
        tape = TracerTape()
        traced_graph = tape.start_tracing("random_creation_trace")
        try:
            r_sym = ensure_tensor(rand(3, 4))
            rn_sym = ensure_tensor(randn(3, 4))
            ri_sym = ensure_tensor(randint(0, 5, (2, 2)))
            seed_out = manual_seed(99)
        finally:
            tape.stop_tracing()

    assert r_sym.shape == (3, 4)
    assert rn_sym.shape == (3, 4)
    assert ri_sym.shape == (2, 2)
    assert seed_out == 99

    op_types = {node.op_type for node in traced_graph.nodes.values()}
    assert {"Rand", "Randn", "Randint", "ManualSeed"} <= op_types
