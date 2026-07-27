import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.io_ops as mod


def test_io_ops_coverage(tmp_path):
    class DummyBackend:
        pass

    bk = DummyBackend()

    # save/load
    p = tmp_path / "f.npy"
    mod._np_save(bk, str(p), np.array([1.0]))
    mod._np_save(bk, filepath=str(p), arr=np.array([1.0]))
    mod._np_save(bk, file=str(p), arr=np.array([1.0]))

    with pytest.raises(ValueError):
        mod._np_load(bk, str(p))

    # savez
    pz = tmp_path / "fz.npz"
    mod._np_savez(bk, str(pz), np.array([1.0]), a=np.array([2.0]))
    mod._np_savez(bk, file=str(pz), args=[np.array([1.0])], kwds={"a": np.array([2.0])})

    mod._np_savez_compressed(bk, str(pz), np.array([1.0]), a=np.array([2.0]))
    mod._np_savez_compressed(bk, file=str(pz), args=[np.array([1.0])], kwds={"a": np.array([2.0])})

    res = mod._np_load(bk, str(pz))
    assert res is not None
    res2 = mod._np_load(bk, file=str(pz))

    # file io
    p_file = tmp_path / "f.txt"
    mod._np_write_file(bk, str(p_file), b"hello")
    mod._np_write_file(bk, filename=str(p_file), contents=b"hello")
    mod._np_write_file(bk, 123, b"hello")  # invalid

    res_file = mod._np_read_file(bk, str(p_file))
    res_file2 = mod._np_read_file(bk, filename=str(p_file))
    mod._np_read_file(bk, 123)

    # decode img
    mod._np_decode_image(bk, b"not image")
    mod._np_decode_image(bk, contents=b"not image")
    mod._np_decode_image(bk, 123)  # not bytes

    # decode csv
    mod._np_decode_csv(bk, [b"1,2", b"3,4"])
    mod._np_decode_csv(bk, records=[b"1,2", b"3,4"], record_defaults=[[0.0], [0.0]])

    # tf examples (dummies)
    mod._np_parse_example(bk, b"")
    mod._np_parse_example(bk, serialized=b"", features={"f": object()})

    # sequence example
    mod._np_parse_sequence_example(bk, b"")
    mod._np_parse_sequence_example(bk, serialized=b"")
    mod._np_parse_sequence_example(bk, 123)  # not bytes

    # base64
    res_b64 = mod._np_encode_base64(bk, b"hello")
    res_b64_2 = mod._np_encode_base64(bk, input=b"hello")

    mod._np_decode_base64(bk, b"aGVsbG8=")
    mod._np_decode_base64(bk, input=b"aGVsbG8=")

    mod._np_encode_base64(bk, None)
    mod._np_decode_base64(bk, None)

    # tensor serialize
    mod._np_serialize_tensor(bk, np.array([1.0]))
    mod._np_serialize_tensor(bk, tensor=np.array([1.0]))

    mod._np_parse_tensor(bk, b"")
    mod._np_parse_tensor(bk, serialized=b"")

    with pytest.raises(RuntimeError):
        mod._np_save_gguf(bk, str(tmp_path / "f.gguf"), {"a": np.array([1.0])})

    with pytest.raises(RuntimeError):
        mod._np_save_gguf(bk, filepath=str(tmp_path / "f.gguf"), tensors={"a": np.array([1.0])})

    mod._np_parse_sequence_example(bk, None)
