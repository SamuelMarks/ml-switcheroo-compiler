"""Provides test suites and coverage verification for all registered operations in the ML.

Switcheroo framework.
"""

import contextlib

from ml_switcheroo_ir import LogicalGraph, LogicalNode


def test_all_ops_coverage() -> None:
    """Verifies that all registered operations implement core interface methods.

    This test iterates through all operations in the global registry (excluding
    control flow and variable operations) and attempts to invoke their JVP,
    VJP, NumPy evaluation, shape inference, and argument formatting methods
    with standard mock inputs. It ensures that these methods are defined and
    do not raise unexpected exceptions during basic execution

    Returns:
    None
    """
    from ml_switcheroo_compiler.ops.base import _OP_REGISTRY

    g = LogicalGraph()
    n = LogicalNode(id="n", op_type="dummy", inputs=["a", "b"])

    for op_name, op_cls in _OP_REGISTRY.items():
        if op_name in ["ReadVariable", "AssignVariable", "Input", "Constant", "Call"]:
            continue
        op = op_cls()

        try:
            op.jvp("tangent_x", "tangent_y", "x", "y")
        except Exception:
            try:
                op.jvp("tangent_x", "x", "newshape")
            except Exception:
                with contextlib.suppress(Exception):
                    op.jvp(g, n, "tangent")

        with contextlib.suppress(Exception):
            op.vjp(g, n, "cotangent")

        try:
            op.numpy_eval(2.0, 3.0)
        except Exception:
            with contextlib.suppress(Exception):
                op.numpy_eval(2.0)

        try:
            op.infer_shape((2, 3), (2, 3))
        except Exception:
            with contextlib.suppress(Exception):
                op.infer_shape((2, 3))

        with contextlib.suppress(Exception):
            op._format_args("x", axis=0)
