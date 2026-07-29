import os
import tempfile

import numpy as np


def test_io_reading_writing():
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import load, read_file, save, savez, write_file

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph") as graph:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "test.txt")
                assert isinstance(write_file(path, "hello world"), Tensor)
                assert isinstance(read_file(path), Tensor)

                path_npy = os.path.join(d, "test.npy")
                t = Tensor(np.array([1, 2, 3]), TensorConfig((3,), "int64", "cpu"))
                assert isinstance(save(path_npy, t), Tensor)

                assert isinstance(load(path_npy), Tensor)

                path_npz = os.path.join(d, "test.npz")
                assert isinstance(savez(path_npz, t=t), Tensor)


def test_io_encoding():
    import unittest.mock as mock

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ops.io import decode_base64, encode_base64

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph") as graph:
            t = Tensor(np.array([b"hello"]), TensorConfig((1,), "string", "cpu"))
            encoded = encode_base64(t)
            assert isinstance(encoded, Tensor)
            decoded = decode_base64(encoded)
            assert isinstance(decoded, Tensor)
