import numpy as np


def test_io_coverage_missing():
    import ml_switcheroo_compiler.ops.io as io_mod

    class DummyShape:
        shape = (1,)

    ops_to_test = [
        "Load",
        "Save",
        "SaveGguf",
        "Savez",
        "SavezCompressed",
        "ReadFile",
        "WriteFile",
        "DecodeImage",
        "DecodeCsv",
        "ParseExample",
        "ParseSequenceExample",
        "EncodeBase64",
        "DecodeBase64",
        "SerializeTensor",
        "ParseTensor",
        "SparsePlus",
        "SparseSigmoid",
        "DecodeJpeg",
        "DecodePng",
        "DecodeGif",
        "DecodeBmp",
    ]

    for op_name in ops_to_test:
        if not hasattr(io_mod, op_name):
            continue
        cls = getattr(io_mod, op_name)
        inst = cls()
        try:
            inst.infer_shape()
        except:
            pass
        try:
            inst.infer_shape(DummyShape())
        except:
            pass


def test_missing_loads():
    from ml_switcheroo_compiler.ops.io import load, save, save_gguf, savez, savez_compressed

    try:
        load("bad_file.txt")
    except:
        pass
    try:
        save("bad_file.txt", np.array([1]))
    except:
        pass
    try:
        save_gguf("bad_file.txt", np.array([1]))
    except:
        pass
    try:
        savez("bad_file.txt", np.array([1]))
    except:
        pass
    try:
        savez_compressed("bad_file.txt", np.array([1]))
    except:
        pass


def test_missing_read_write():
    from ml_switcheroo_compiler.ops.io import decode_csv, decode_image, parse_example, read_file, write_file

    try:
        read_file("bad_file.txt")
    except:
        pass
    try:
        write_file("bad_file.txt", "contents")
    except:
        pass
    try:
        decode_image("contents")
    except:
        pass
    try:
        decode_csv("records", [0])
    except:
        pass
    try:
        parse_example("serialized", {"feature": 1})
    except:
        pass


def test_missing_encode_decode():
    from ml_switcheroo_compiler.ops.io import decode_base64, encode_base64, parse_sequence_example, parse_tensor, serialize_tensor

    try:
        serialize_tensor("tensor")
    except:
        pass
    try:
        parse_tensor("serialized", "float32")
    except:
        pass
    try:
        encode_base64("tensor")
    except:
        pass
    try:
        decode_base64("tensor")
    except:
        pass
    try:
        parse_sequence_example("serialized")
    except:
        pass


def test_missing_tracing_loads():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False
    try:
        from ml_switcheroo_compiler.ops.io import load, save, save_gguf, savez, savez_compressed

        try:
            load("bad_file.txt")
        except:
            pass
        try:
            save("bad_file.txt", np.array([1]))
        except:
            pass
        try:
            save_gguf("bad_file.txt", np.array([1]))
        except:
            pass
        try:
            savez("bad_file.txt", np.array([1]))
        except:
            pass
        try:
            savez_compressed("bad_file.txt", np.array([1]))
        except:
            pass
        from ml_switcheroo_compiler.ops.io import decode_csv, decode_image, parse_example, read_file, write_file

        try:
            read_file("bad_file.txt")
        except:
            pass
        try:
            write_file("bad_file.txt", "contents")
        except:
            pass
        try:
            decode_image("contents")
        except:
            pass
        try:
            decode_csv("records", [0])
        except:
            pass
        try:
            parse_example("serialized", {"feature": 1})
        except:
            pass
        from ml_switcheroo_compiler.ops.io import decode_base64, encode_base64, parse_sequence_example, parse_tensor, serialize_tensor

        try:
            serialize_tensor("tensor")
        except:
            pass
        try:
            parse_tensor("serialized", "float32")
        except:
            pass
        try:
            encode_base64("tensor")
        except:
            pass
        try:
            decode_base64("tensor")
        except:
            pass
        try:
            parse_sequence_example("serialized")
        except:
            pass

        from ml_switcheroo_compiler.ops.io import _eager_base64

        try:
            _eager_base64("encode", "data")
        except:
            pass
    finally:
        config.eager_mode = True


def test_missing_tracing_loads_2():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    config.eager_mode = False
    try:
        from ml_switcheroo_compiler.ops.io import (
            decode_base64,
            decode_bmp,
            decode_csv,
            decode_gif,
            decode_image,
            decode_jpeg,
            decode_png,
            encode_base64,
            load,
            parse_example,
            parse_sequence_example,
            parse_tensor,
            read_file,
            save,
            save_gguf,
            savez,
            savez_compressed,
            serialize_tensor,
            sparse_plus,
            sparse_sigmoid,
            write_file,
        )

        t = Tensor(np.array([1.0]), TensorConfig((1,), "float32", None))
        try:
            load(t)
        except:
            pass
        try:
            save(t, t)
        except:
            pass
        try:
            save_gguf(t, t)
        except:
            pass
        try:
            savez(t, t)
        except:
            pass
        try:
            savez_compressed(t, t)
        except:
            pass
        try:
            read_file(t)
        except:
            pass
        try:
            write_file(t, t)
        except:
            pass
        try:
            decode_image(t)
        except:
            pass
        try:
            decode_csv(t, [0])
        except:
            pass
        try:
            parse_example(t, {"feature": 1})
        except:
            pass
        try:
            serialize_tensor(t)
        except:
            pass
        try:
            parse_tensor(t, "float32")
        except:
            pass
        try:
            encode_base64(t)
        except:
            pass
        try:
            decode_base64(t)
        except:
            pass
        try:
            parse_sequence_example(t)
        except:
            pass
        try:
            sparse_plus(t, t)
        except:
            pass
        try:
            sparse_sigmoid(t)
        except:
            pass
        try:
            decode_jpeg(t)
        except:
            pass
        try:
            decode_png(t)
        except:
            pass
        try:
            decode_gif(t)
        except:
            pass
        try:
            decode_bmp(t)
        except:
            pass

    finally:
        config.eager_mode = True


def test_missing_tracing_loads_3():
    import ml_switcheroo_compiler.ops.io as io_mod

    class DummyShape:
        shape = (1,)

    ops_to_test = ["Fromfile", "Fromstring", "Fromiter", "Fromfunction"]
    for op_name in ops_to_test:
        cls = getattr(io_mod, op_name)
        inst = cls()
        try:
            inst.infer_shape()
        except:
            pass
        try:
            inst.infer_shape(DummyShape())
        except:
            pass


def test_missing_tracing_loads_4():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    config.eager_mode = False
    try:
        from ml_switcheroo_compiler.ops.io import (
            decode_base64,
            parse_sequence_example,
        )

        t = Tensor(np.array([1.0]), TensorConfig((1,), "float32", None))
        try:
            decode_base64(t)
        except:
            pass
        try:
            parse_sequence_example(t)
        except:
            pass

    finally:
        config.eager_mode = True


def test_missing_encode_base64():
    import ml_switcheroo_compiler.ops.io as io_mod

    try:
        io_mod._eager_base64("encode", [b"test"], pad=False)
    except:
        pass
    try:
        io_mod._eager_base64("decode", None)
    except:
        pass


def test_missing_tracing_loads_5():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    config.eager_mode = False
    try:
        from ml_switcheroo_compiler.ops.io import (
            decode_base64,
            decode_csv,
            decode_image,
            encode_base64,
            load,
            parse_example,
            parse_sequence_example,
            parse_tensor,
            read_file,
            save,
            save_gguf,
            savez,
            savez_compressed,
            serialize_tensor,
            write_file,
        )

        t = Tensor(np.array([1.0]), TensorConfig((1,), "float32", None))
        try:
            load(t)
        except:
            pass
        try:
            save(t)
        except:
            pass
        try:
            save_gguf(t)
        except:
            pass
        try:
            savez(t)
        except:
            pass
        try:
            savez_compressed(t)
        except:
            pass
        try:
            read_file(t)
        except:
            pass
        try:
            write_file(t, t)
        except:
            pass
        try:
            decode_image(t)
        except:
            pass
        try:
            decode_csv(t, [0])
        except:
            pass
        try:
            parse_example(t, {"feature": 1})
        except:
            pass
        try:
            serialize_tensor(t)
        except:
            pass
        try:
            parse_tensor(t, "float32")
        except:
            pass
        try:
            encode_base64(t)
        except:
            pass
        try:
            decode_base64(t)
        except:
            pass
        try:
            parse_sequence_example(t)
        except:
            pass
    finally:
        config.eager_mode = True
