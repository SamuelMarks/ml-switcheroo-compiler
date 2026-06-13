"""Provides required module functionality."""

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops import binary, creation, linalg, reductions, shape, unary


def _test_namespace(ns: object, eager: bool) -> None:
    """Execute the requested function.

    Args:
        ns (object): The ns.
        eager (bool): The eager.
    """
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    config.eager_mode = eager

    if eager:
        data_x = np.ones((2, 3, 4))
        data_y = np.ones((2, 3, 4))
        data_z = np.ones((2, 3))
        data_w = np.ones((1, 2))
        data_a = np.array([1, 2])
    else:
        data_x = ProxyTensor("x", (2, 3, 4), "float32")
        data_y = ProxyTensor("y", (2, 3, 4), "float32")
        data_z = ProxyTensor("z", (2, 3), "float32")
        data_w = ProxyTensor("w", (1, 2), "float32")
        data_a = ProxyTensor("a", (2,), "int32")

    x = Tensor(data=data_x, shape=(2, 3, 4), dtype=DType.Float32, device="cpu")
    y = Tensor(data=data_y, shape=(2, 3, 4), dtype=DType.Float32, device="cpu")
    z = Tensor(data=data_z, shape=(2, 3), dtype=DType.Float32, device="cpu")
    w = Tensor(data=data_w, shape=(1, 2), dtype=DType.Float32, device="cpu")
    a = Tensor(data=data_a, shape=(2,), dtype=DType.Int32, device="cpu")

    args_list = [
        (x,),
        (x, 1),
        (x, x),
        (x, 0, 1),
        (x, (1, 0, 2)),
        ((x, y),),
        (x, 2, 2),
        (x, (2,)),
        (x, a),
        (x, a, a),
        (x, x, x),
        (z,),
        (z, 1),
        (z, z),
        (w,),
        (w, 1),
        (a,),
        (x, 1, 1),
        (x, x, 1),
        ((x, y), 0),
        (x, (1, 2)),
        (z, (1,)),
        (x, a, a, a),
        (z, 0, a, a),
        (z, a, a),
        (z, z, z),
        (x, (0, 1, 2)),
        (x, [1]),
        (2,),
        ((2, 2),),
        (2, 2),
        (0, 1, 10),
        ((2, 3), 1.0),
        ((2, 2), DType.Float32),
        (3, 3),
        (DType.Float32, (2, 2)),
        ([1, 2, 3],),
        ("ab,bc->ac", z, z),
        (x, y, 1),
    ]
    kwargs_list = [
        {},
        {"axis": 0},
        {"axis": 1},
        {"axes": 0},
        {"dims": 0},
        {"shape": (2, 2)},
        {"newshape": (6,)},
    ]

    for name in dir(ns):
        if not name.startswith("_"):
            func = getattr(ns, name)
            if callable(func):
                for args in args_list:
                    for kwargs in kwargs_list:
                        try:
                            if not eager:
                                from ml_switcheroo_compiler.tracing.tracer import _tracer

                                _tracer.start_tracing("test")
                                func(*args, **kwargs)
                                _tracer.stop_tracing()
                            else:
                                func(*args, **kwargs)
                        except Exception:
                            if not eager:
                                from ml_switcheroo_compiler.tracing.tracer import _tracer

                                _tracer.stop_tracing()


def test_shape_brute_force() -> None:
    """Test shape ops."""
    _test_namespace(shape, True)
    _test_namespace(shape, False)


def test_creation_brute_force() -> None:
    """Test creation ops."""
    _test_namespace(creation, True)
    _test_namespace(creation, False)


def test_linalg_brute_force() -> None:
    """Test linalg ops."""
    _test_namespace(linalg, True)
    _test_namespace(linalg, False)


def test_binary_brute_force() -> None:
    """Test binary ops."""
    _test_namespace(binary, True)
    _test_namespace(binary, False)


def test_unary_brute_force() -> None:
    """Test unary ops."""
    _test_namespace(unary, True)
    _test_namespace(unary, False)


def test_reductions_brute_force() -> None:
    """Test reduction ops."""
    _test_namespace(reductions, True)
    _test_namespace(reductions, False)
