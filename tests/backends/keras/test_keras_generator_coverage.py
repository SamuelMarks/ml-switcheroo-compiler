"""Test Keras generator edge cases coverage."""

from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator


def test_keras_generator_save_load(tmp_path):
    """Test save and load methods of Keras generator."""
    arr = [1, 2, 3]
    filepath = str(tmp_path / "dummy.pkl")

    # save
    KerasCodeGenerator.save(filepath, arr)

    # load
    res = KerasCodeGenerator.load(filepath)
    assert res == arr

    # savez
    filez = str(tmp_path / "dummy.npz")
    KerasCodeGenerator.savez(filez, arr, arg2=[4, 5])

    resz = KerasCodeGenerator.load(filez)
    assert resz["arr_0"] == arr
    assert resz["arg2"] == [4, 5]

    # savez_compressed
    filezc = str(tmp_path / "dummy_comp.npz")
    KerasCodeGenerator.savez_compressed(filezc, arr, arg2=[4, 5])

    import gzip
    import pickle

    with gzip.open(filezc, "rb") as f:
        reszc = pickle.load(f)
    assert reszc["arr_0"] == arr
    assert reszc["arg2"] == [4, 5]


def test_keras_generator_conv_transpose():
    from ml_switcheroo_ir import LogicalGraph

    gen = KerasCodeGenerator(LogicalGraph())

    class DummyNode:
        pass

    assert gen.visit_ConvTranspose(DummyNode(), ["inp1", "inp2"]) == "keras_conv_transpose(inp1, inp2)"
    assert gen.visit_RaggedDot(DummyNode(), ["inp1", "inp2"]) == "keras_ragged_dot(inp1, inp2)"
