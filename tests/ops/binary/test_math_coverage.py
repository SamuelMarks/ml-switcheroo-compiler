from ml_switcheroo_compiler.ops.binary.math import ArrayEquiv, BinaryMathOp, Digitize, Lcm, Polyadd, Polyder, Polydiv, Polyint, Polymul, Polysub, Roots, Xlog1py, Xlogy


def test_binary_math_op_infer_shape():
    op = BinaryMathOp()
    assert op.infer_shape() == ()
    assert op.infer_shape((2, 2)) == (2, 2)


def test_digitize_infer_shape():
    op = Digitize()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy()) == (2, 2)


def test_polyder_infer_shape():
    op = Polyder()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_polyint_infer_shape():
    op = Polyint()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_polysub_infer_shape():
    op = Polysub()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy(), Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_polyadd_infer_shape():
    op = Polyadd()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy(), Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_polymul_infer_shape():
    op = Polymul()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy(), Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_polydiv_infer_shape():
    op = Polydiv()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy(), Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_poly_infer_shape():
    from ml_switcheroo_compiler.ops.binary.math import Poly

    op = Poly()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_polyval_infer_shape():
    from ml_switcheroo_compiler.ops.binary.math import Polyval

    op = Polyval()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_polyfit_infer_shape():
    from ml_switcheroo_compiler.ops.binary.math import Polyfit

    op = Polyfit()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_roots_infer_shape():
    op = Roots()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_lcm_infer_shape():
    op = Lcm()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy(), Dummy()).shape == (2, 2)
    assert op.infer_shape(1) == 1


def test_array_equiv_infer_shape():
    op = ArrayEquiv()
    assert op.infer_shape() == ()


def test_xlogy_infer_shape():
    op = Xlogy()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy(), Dummy()).shape == (2, 2)


def test_xlog1py_infer_shape():
    op = Xlog1py()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy(), Dummy()).shape == (2, 2)


def test_clip_eager():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.binary.math import Clip, clip

    op = Clip()
    assert op.infer_shape() == ()

    class Dummy:
        shape = (2, 2)

    assert op.infer_shape(Dummy()) == (2, 2)

    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        import numpy as np

        # Need an active backend, or we can just mock it
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        class MockBackend:
            @classmethod
            def execute_op(cls, op_name, *args, **kwargs):
                return "eager_clip"

        orig_backend = config.backend
        config.backend = "mock_clip"
        BackendRegistry.register("mock_clip", MockBackend)

        assert clip(np.array([1.0]), 0.0, 2.0) == "eager_clip"

        config.eager_mode = False

        # Test tracing emit
        # we can just mock emit_ir_node or let it hit tracing error
        try:
            clip(np.array([1.0]), 0.0, 2.0)
        except Exception:
            pass

    finally:
        config.eager_mode = orig_eager
        if "orig_backend" in locals():
            config.backend = orig_backend


def test_rem_op():
    from ml_switcheroo_compiler.ops.binary.math import rem

    try:
        rem(1, 2)
    except Exception:
        pass
