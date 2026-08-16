import os
import struct

from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_export_mlirbc(tmp_path):
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    g.nodes = {"n1": n1}

    gen = StableHLOCodeGenerator(g)
    out_file = os.path.join(tmp_path, "model.mlirbc")

    gen.export_mlirbc(out_file)

    assert os.path.exists(out_file)
    with open(out_file, "rb") as f:
        magic = f.read(4)
        assert magic == b"ML\xefR"  # 4D 4C EF 52

        version = struct.unpack("<B", f.read(1))[0]
        assert version == 1

        # Parse payload string sections
        # First byte is producer string ID
        f.read(1)
        # Then section ID
        sec_id = struct.unpack("<B", f.read(1))[0]
        assert sec_id == 0  # SECTION_STRING

        # We can just read the rest of the payload and check if our strings are in it
        payload = f.read()

        # Test it contains stablehlo
        assert b"stablehlo" in payload
        assert b"ml_switcheroo_compiler" in payload

    def test_mlir_bytecode_encoder_coverage():
        from ml_switcheroo_compiler.backends.edge.mlir_bytecode import MLIRBytecodeEncoder

        enc = MLIRBytecodeEncoder()

        # Test large varint to cover line 52
        b = enc._encode_varint(300)
        assert len(b) > 1

        # Test add_op
        enc.add_op("my_op", ["arg1"], ["ret1"])
        assert len(enc.ops) == 1

        # Test add_dialect duplicate
        enc.add_dialect("func")
        enc.add_dialect("func")

        out = enc.encode()
        assert b"my_op" in out
